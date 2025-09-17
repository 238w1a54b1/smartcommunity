from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Profile, Tag, Thread, Comment

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'follower_count', 'following_count')

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ('name',)

@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
    list_display = ('id', 'author', 'created_at', 'privacy', 'summary')
    list_filter = ('privacy', 'created_at')
    search_fields = ('content', 'author__username')

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ('id', 'thread', 'author', 'created_at')
    search_fields = ('content', 'author__username')