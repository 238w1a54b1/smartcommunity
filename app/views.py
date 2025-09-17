from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Q, Count
from django.core.paginator import Paginator
from django.conf import settings

# External libraries for API calls and math
import requests
import numpy as np

# Channels for real-time updates
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

# Local app imports
from .forms import ProfileUpdateForm, ThreadCreateForm, CommentForm, CommentEditForm
from .models import Profile, Thread, Comment, Tag


# ==============================================================================
# API HELPER FUNCTIONS
# ==============================================================================

def generate_summary_hf_api(text, min_length=20, max_length=60):
    """Generate a summary using Hugging Face Inference API."""
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    token = settings.HUGGINGFACE_API_TOKEN
    if not token:
        return ""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {"min_length": min_length, "max_length": max_length}
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
    except requests.exceptions.RequestException as e:
        print(f"HuggingFace summary API request failed: {e}")
    return ""

def is_toxic_text(text: str, threshold: float = 0.7) -> bool:
    """Simple toxicity check using HF inference API (unitary/toxic-bert)."""
    token = settings.HUGGINGFACE_API_TOKEN
    if not token:
        return False
    api_url = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                scores = {item.get("label"): item.get("score", 0.0) for item in data[0] if isinstance(item, dict)}
                # Consider any high-scoring toxic label as toxic
                toxic_score = max(scores.values()) if scores else 0.0
                return toxic_score >= threshold
    except requests.exceptions.RequestException as e:
        print(f"HuggingFace toxicity API request failed: {e}")
        return False
    return False

def extract_tags_api(text, top_n=5):
    """Extract tags using Hugging Face Inference API."""
    token = settings.HUGGINGFACE_API_TOKEN
    if not token:
        return []
    API_URL = "https://api-inference.huggingface.co/models/ml6team/keyphrase-extraction-distilbert-inspec"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": text}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and len(results) > 0:
                keywords = [r.get("word") for r in results[:top_n] if isinstance(r, dict)]
                return [k for k in keywords if isinstance(k, str)]
    except requests.exceptions.RequestException as e:
        print(f"HuggingFace tags API request failed: {e}")
    return []


# ==============================================================================
# DJANGO VIEWS
# ==============================================================================

@login_required
def home(request):
    latest_threads = Thread.objects.select_related('author').prefetch_related('tags').order_by('-created_at')[:10]
    trending_tags = Tag.objects.annotate(thread_count=Count('threads')).order_by('-thread_count')[:8]
    suggested_users = User.objects.exclude(id=request.user.id)[:6]
    return render(request, "comm.html", {
        'latest_threads': latest_threads,
        'trending_tags': trending_tags,
        'suggested_users': suggested_users,
    })

def signup(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password_1 = request.POST.get("password")
        password_2 = request.POST.get("confirm_password")
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")

        if password_1 != password_2:
            messages.error(request, "Passwords do not match!")
            return redirect("signup")
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
            return redirect("signup")
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            return redirect("signup")

        user = User.objects.create_user(username=username, email=email, password=password_1, first_name=first_name, last_name=last_name)
        user.save()
        messages.success(request, "Account created successfully! Please log in.")
        return redirect("signin")
    return render(request, "signup.html")

def signin(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        try:
            user_obj = User.objects.get(email=email)
            user = authenticate(request, username=user_obj.username, password=password)
        except User.DoesNotExist:
            user = None

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            messages.error(request, "Invalid email or password")
            return redirect("signin")
    return render(request, "signin.html")

def signout(request):
    logout(request)
    return redirect("home")

@login_required
def view_profile(request):
    current_user_following = request.user.following.all()
    suggested_users = User.objects.exclude(id=request.user.id).exclude(profile__in=current_user_following)[:5]
    threads = Thread.objects.filter(author=request.user).order_by('-created_at')[:10]
    post_count = Thread.objects.filter(author=request.user).count()
    reply_count = Comment.objects.filter(author=request.user).count()
    profile, created = Profile.objects.get_or_create(user=request.user)
    points = post_count * 5 + reply_count * 2 + profile.follower_count()
    
    badges = []
    if post_count >= 1: badges.append({'name': 'First Post'})
    if post_count >= 10: badges.append({'name': 'Active Poster'})
    if reply_count >= 25: badges.append({'name': 'Conversationalist'})
    if profile.follower_count() >= 10: badges.append({'name': 'Popular'})
    
    context = {
        'suggested_users': suggested_users, 'threads': threads,
        'post_count': post_count, 'reply_count': reply_count,
        'points': points, 'badges': badges,
    }
    return render(request, "myprofile.html", context)

@login_required
def edit_profile(request):
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect('edit_profile')
    else:
        form = ProfileUpdateForm(instance=request.user.profile)
    return render(request, 'edit_profile.html', {'form': form})

@login_required
def follow_toggle(request, user_id):
    target_user = get_object_or_404(User, id=user_id)
    if target_user != request.user:
        target_profile, _ = Profile.objects.get_or_create(user=target_user)
        if request.user in target_profile.followers.all():
            target_profile.followers.remove(request.user)
            messages.success(request, f"Unfollowed {target_user.username}.")
        else:
            target_profile.followers.add(request.user)
            messages.success(request, f"Now following {target_user.username}.")
    return redirect(request.META.get('HTTP_REFERER', 'view_profile'))

@login_required
def followers_list(request, user_id):
    target_user = get_object_or_404(User, id=user_id)
    target_profile, _ = Profile.objects.get_or_create(user=target_user)
    followers_qs = target_profile.followers.select_related('profile').all()
    current_user_following = request.user.following.all()
    suggested_users = User.objects.exclude(id=request.user.id).exclude(profile__in=current_user_following)[:5]
    context = {'target_user': target_user, 'followers': followers_qs, 'suggested_users': suggested_users}
    return render(request, 'followers_list.html', context)

@login_required
def following_list(request, user_id):
    target_user = get_object_or_404(User, id=user_id)
    following_profiles = target_user.following.select_related('user').all()
    current_user_following = request.user.following.all()
    suggested_users = User.objects.exclude(id=request.user.id).exclude(profile__in=current_user_following)[:5]
    context = {'target_user': target_user, 'following_profiles': following_profiles, 'suggested_users': suggested_users}
    return render(request, 'following_list.html', context)

@login_required
def threads(request):
    Profile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = ThreadCreateForm(request.POST, request.FILES)
        if form.is_valid():
            thread = form.save(commit=False)
            thread.author = request.user
            if is_toxic_text(thread.content):
                messages.error(request, 'Your post appears toxic and was blocked by moderation.')
                return redirect('newthreads')

            # Generate summary and tags using APIs
            thread.summary = generate_summary_hf_api(thread.content)
            thread.save() # Save thread to get an ID
            
            tags_list = extract_tags_api(thread.content)
            for tag_name in tags_list:
                tag, _ = Tag.objects.get_or_create(name=tag_name)
                thread.tags.add(tag)
            
            messages.success(request, 'Thread posted!')
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "threads_stream",
                {
                    "type": "thread_event",
                    "payload": {
                        "event": "thread_created", "thread_id": thread.id,
                        "author": thread.author.username, "summary": thread.summary or "",
                    },
                },
            )
            return redirect('newthreads')
    else:
        form = ThreadCreateForm()

    search_query = request.GET.get('q', '')
    tag_filter = request.GET.get('tag', '')
    privacy_filter = request.GET.get('privacy', '')
    sort_by = request.GET.get('sort', 'newest')
    
    threads_qs = Thread.objects.select_related('author').prefetch_related('tags', 'likes', 'saves', 'comments').all()
    
    if search_query:
        threads_qs = threads_qs.filter(
            Q(content__icontains=search_query) | Q(author__username__icontains=search_query) |
            Q(tags__name__icontains=search_query)
        ).distinct()
    if tag_filter:
        threads_qs = threads_qs.filter(tags__name=tag_filter)
    if privacy_filter:
        threads_qs = threads_qs.filter(privacy=privacy_filter)
    
    if sort_by == 'popular':
        threads_qs = threads_qs.annotate(like_count_sort=Count('likes')).order_by('-like_count_sort', '-created_at')
    elif sort_by == 'commented':
        threads_qs = threads_qs.annotate(comment_count_sort=Count('comments')).order_by('-comment_count_sort', '-created_at')
    else:
        threads_qs = threads_qs.order_by('-created_at')
    
    visible_threads = [th for th in threads_qs if th.can_view(request.user)]
    
    paginator = Paginator(visible_threads, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    current_user_following = request.user.following.all()
    suggested_users = User.objects.exclude(id=request.user.id).exclude(profile__in=current_user_following)[:6]
    trending_tags = Tag.objects.annotate(thread_count=Count('threads')).filter(thread_count__gt=0).order_by('-thread_count')[:10]

    context = {
        'suggested_users': suggested_users, 'trending_tags': trending_tags,
        'threads': page_obj, 'form': form, 'comment_form': CommentForm(),
        'search_query': search_query, 'tag_filter': tag_filter,
        'privacy_filter': privacy_filter, 'sort_by': sort_by,
        'privacy_choices': Thread.Privacy.choices,
    }
    return render(request, 'threads.html', context)

@login_required
def saved_threads(request):
    saved_threads_qs = Thread.objects.filter(saves=request.user).select_related('author').prefetch_related('tags', 'likes', 'comments').order_by('-created_at')
    paginator = Paginator(saved_threads_qs, 10)
    page_obj = paginator.get_page(request.GET.get('page'))
    context = {'threads': page_obj, 'comment_form': CommentForm()}
    return render(request, 'saved_threads.html', context)

@login_required
def tag_threads(request, tag_name):
    tag = get_object_or_404(Tag, name=tag_name)
    threads_qs = Thread.objects.filter(tags=tag).select_related('author').prefetch_related('tags', 'likes', 'comments').order_by('-created_at')
    visible_threads = [th for th in threads_qs if th.can_view(request.user)]
    paginator = Paginator(visible_threads, 10)
    page_obj = paginator.get_page(request.GET.get('page'))
    context = {'tag': tag, 'threads': page_obj, 'comment_form': CommentForm()}
    return render(request, 'tag_threads.html', context)

@login_required
def search_threads(request):
    query = request.GET.get('q', '')
    search_type = request.GET.get('type', 'all')
    results = {'threads': [], 'users': [], 'tags': []}
    if query:
        if search_type in ['all', 'threads']:
            threads_qs = Thread.objects.filter(Q(content__icontains=query) | Q(tags__name__icontains=query)).distinct()
            results['threads'] = [th for th in threads_qs if th.can_view(request.user)][:10]
        if search_type in ['all', 'users']:
            results['users'] = User.objects.filter(Q(username__icontains=query) | Q(first_name__icontains=query) | Q(last_name__icontains=query))[:10]
        if search_type in ['all', 'tags']:
            results['tags'] = Tag.objects.filter(name__icontains=query)[:10]
    context = {'query': query, 'search_type': search_type, 'results': results}
    return render(request, 'search_results.html', context)

@login_required
def explore(request):
    top_tags = Tag.objects.annotate(thread_count=Count('threads')).order_by('-thread_count')[:20]
    recent_threads = Thread.objects.select_related('author').prefetch_related('tags').order_by('-created_at')[:30]
    suggested_users = User.objects.exclude(id=request.user.id)[:10]
    return render(request, 'explore.html', {'top_tags': top_tags, 'recent_threads': recent_threads, 'suggested_users': suggested_users})

@login_required
def leaderboard(request):
    leaderboard_data = User.objects.annotate(
        posts_count=Count('threads', distinct=True),
        replies_count=Count('comments', distinct=True)
    ).values('id', 'username', 'first_name', 'last_name', 'posts_count', 'replies_count')
    
    ranked = [{'points': u['posts_count'] * 5 + u['replies_count'] * 2, **u} for u in leaderboard_data]
    ranked.sort(key=lambda x: x['points'], reverse=True)
    return render(request, 'leaderboard.html', {'ranked': ranked[:50]})

@login_required
def similar_threads(request, thread_id):
    """Return JSON with similar threads using Hugging Face Feature Extraction API."""
    base_thread = get_object_or_404(Thread, id=thread_id)
    candidates = Thread.objects.exclude(id=thread_id).order_by('-created_at')[:50]
    
    token = settings.HUGGINGFACE_API_TOKEN
    if not token or not candidates:
        return JsonResponse({'items': []})

    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-MiniLM-L3-v2"
    headers = {"Authorization": f"Bearer {token}"}
    texts_to_embed = [base_thread.content] + [t.content for t in candidates]
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": texts_to_embed, "options": {"wait_for_model": True}}, timeout=45)
        if response.status_code != 200:
            return JsonResponse({'items': []}, status=500)

        embeddings = response.json()
        if not isinstance(embeddings, list) or len(embeddings) != len(texts_to_embed):
            return JsonResponse({'items': []}, status=500)
        
        base_vec = np.array(embeddings[0])
        cand_vecs = np.array(embeddings[1:])
        base_vec_norm = base_vec / np.linalg.norm(base_vec)
        cand_vecs_norm = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)

        sims = np.dot(cand_vecs_norm, base_vec_norm)
        top_indices = np.argsort(-sims)[:5]
        
        results = [{'id': candidates[idx].id, 'snippet': (candidates[idx].content[:160] + '...'), 'score': float(sims[idx])} for idx in top_indices]
        return JsonResponse({'items': results})
        
    except Exception as e:
        print(f"Error processing similar threads with API: {e}")
        return JsonResponse({'items': []}, status=500)

# AJAX and Real-time Views
@login_required
def thread_like_toggle(request, thread_id):
    thread = get_object_or_404(Thread, id=thread_id)
    if request.user in thread.likes.all():
        thread.likes.remove(request.user)
    else:
        thread.likes.add(request.user)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)("threads_stream", {"type": "thread_event", "payload": {"event": "thread_liked", "thread_id": thread.id, "likes": thread.like_count}})
        return JsonResponse({'ok': True, 'likes': thread.like_count})
    return redirect('newthreads')

@login_required
def thread_save_toggle(request, thread_id):
    thread = get_object_or_404(Thread, id=thread_id)
    if request.user in thread.saves.all():
        thread.saves.remove(request.user)
    else:
        thread.saves.add(request.user)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'ok': True})
    return redirect('newthreads')

@login_required
def comment_create(request, thread_id):
    thread = get_object_or_404(Thread, id=thread_id)
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.thread = thread
            comment.author = request.user
            if is_toxic_text(comment.content):
                if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                    return JsonResponse({'ok': False, 'error': 'Toxic content blocked'}, status=400)
                messages.error(request, 'Your comment appears toxic and was blocked by moderation.')
                return redirect('newthreads')
            comment.save()
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.group_send)("threads_stream", {"type": "thread_event", "payload": {"event": "comment_created", "thread_id": thread.id, "comment_id": comment.id}})
                return JsonResponse({'ok': True, 'comment_id': comment.id})
            messages.success(request, 'Comment added')
    return redirect('newthreads')

@login_required
def comment_edit(request, comment_id):
    comment = get_object_or_404(Comment, id=comment_id, author=request.user)
    if request.method == 'POST':
        form = CommentEditForm(request.POST, instance=comment)
        if form.is_valid():
            form.save()
            return JsonResponse({'ok': True, 'content': comment.content})
        return JsonResponse({'ok': False, 'errors': form.errors})
    return JsonResponse({'ok': False}, status=400)

@login_required
def comment_delete(request, comment_id):
    comment = get_object_or_404(Comment, id=comment_id, author=request.user)
    if request.method == 'POST': # Ensure deletion is via POST
        comment.delete()
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'ok': True})
        messages.success(request, 'Comment deleted successfully!')
    return redirect('newthreads')

@login_required
def user_suggestions(request):
    current_user_following = request.user.following.all()
    suggested_users = User.objects.exclude(id=request.user.id).exclude(profile__in=current_user_following)[:10]
    context = {'suggested_users': suggested_users}
    return render(request, 'user_suggestions.html', context)