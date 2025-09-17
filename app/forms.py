# app/forms.py
from django import forms
from .models import Profile
from .models import Thread, Comment, Tag

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['profile_image', 'bio']


class ThreadCreateForm(forms.ModelForm):
    

    class Meta:
        model = Thread
        fields = ['title', 'content', 'image', 'link', 'privacy']
        widgets = {
            'title': forms.TextInput(attrs={
                'placeholder': 'Add a clear, descriptive title',
                'class': (
                    "w-full rounded-lg px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none "
                    "bg-white/90 text-gray-900 placeholder-gray-600 border border-gray-300 "
                    "dark:bg-white/10 dark:text-gray-100 dark:placeholder-gray-400 dark:border-gray-700"
                ),
            }),
            'content': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': "What's on your mind?",
                "class": (
                    "w-full rounded-lg p-3 focus:ring-2 focus:ring-indigo-500 focus:outline-none "
                    "bg-white/90 text-gray-900 placeholder-gray-600 border border-gray-300 "
                    "dark:bg-white/10 dark:text-gray-100 dark:placeholder-gray-400 dark:border-gray-700"
                ),
            }),
            'link': forms.URLInput(attrs={
                "class": (
                    "w-full rounded-lg px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none "
                    "bg-white/90 text-gray-900 placeholder-gray-600 border border-gray-300 "
                    "dark:bg-white/10 dark:text-gray-100 dark:placeholder-gray-400 dark:border-gray-700"
                ),
                "placeholder": "Paste a link here...",
            }),
            'privacy': forms.Select(attrs={
                "class": (
                    "w-full rounded-lg px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none "
                    "bg-white/90 text-gray-900 border border-gray-300 "
                    "dark:bg-white/10 dark:text-gray-100 dark:border-gray-700"
                ),
            }),
        }




class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content', 'parent']
        widgets = {
            'content': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Write a thoughtful comment...',
                'class': (
                    "w-full rounded-2xl p-4 outline-none resize-none focus:ring-2 focus:ring-indigo-500 "
                    "bg-white/90 text-gray-900 placeholder-gray-600 border border-gray-300 "
                    "dark:bg-white/10 dark:text-gray-100 dark:placeholder-gray-400 dark:border-gray-700"
                ),
            }),
            'parent': forms.HiddenInput(),
        }



class CommentEditForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Edit your comment...',
                'class': 'w-full glass rounded-2xl p-4 outline-none resize-none focus:ring-2 focus:ring-indigo-500',
            }),
        }
