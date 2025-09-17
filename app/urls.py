from django.urls import path,include
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("signup/", views.signup, name="signup"),
    path("signin/", views.signin, name="signin"),
    path("signout/", views.signout, name="signout"),
    path('accounts/', include('allauth.urls')),
    path("profile/", views.view_profile, name="view_profile"),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
    path('u/<int:user_id>/follow-toggle/', views.follow_toggle, name='follow_toggle'),
    path('u/<int:user_id>/followers/', views.followers_list, name='followers_list'),
    path('u/<int:user_id>/following/', views.following_list, name='following_list'),
    path('suggestions/', views.user_suggestions, name='user_suggestions'),
    path('newthreads/', views.threads, name='newthreads'),
    path('saved/', views.saved_threads, name='saved_threads'),
    path('search/', views.search_threads, name='search_threads'),
    path('tag/<str:tag_name>/', views.tag_threads, name='tag_threads'),
    path('u/<int:thread_id>/like-toggle/', views.thread_like_toggle, name='thread_like_toggle'),
    path('u/<int:thread_id>/save-toggle/', views.thread_save_toggle, name='thread_save_toggle'),
    path('u/<int:thread_id>/comment/', views.comment_create, name='comment_create'),
    path('comment/<int:comment_id>/edit/', views.comment_edit, name='comment_edit'),
    path('comment/<int:comment_id>/delete/', views.comment_delete, name='comment_delete'),
    path('similar/<int:thread_id>/', views.similar_threads, name='similar_threads'),
    path('explore/', views.explore, name='explore'),
    path('leaderboard/', views.leaderboard, name='leaderboard'),
]
