
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from django.utils import timezone

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_image = models.ImageField(upload_to='profile_photos/', default='profile_photos/default.jpg')
    bio = models.TextField(blank=True)
    followers = models.ManyToManyField(User, related_name='following', blank=True)
    def follower_count(self):
        return self.followers.count()  # Number of users following this profile

    def following_count(self):
        return self.user.following.count()  # Number of profiles this user follows

    def __str__(self):
        return self.user.username
# Create your models here.


class Tag(models.Model):
    name = models.CharField(max_length=40, unique=True)

    def __str__(self) -> str:
        return self.name


class Thread(models.Model):
    class Privacy(models.TextChoices):
        PUBLIC = 'public', 'Public'
        FOLLOWERS = 'followers', 'Followers'
        PRIVATE = 'private', 'Private'

    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='threads')
    title = models.CharField(max_length=200, blank=True, null=True, default='')
    content = models.TextField()
    image = models.ImageField(upload_to='thread_media/', blank=True, null=True)
    link = models.URLField(blank=True)
    privacy = models.CharField(max_length=16, choices=Privacy.choices, default=Privacy.PUBLIC)
    tags = models.ManyToManyField(Tag, blank=True, related_name='threads')
    likes = models.ManyToManyField(User, blank=True, related_name='liked_threads')
    saves = models.ManyToManyField(User, blank=True, related_name='saved_threads')
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    summary = models.TextField(blank=True, null=True)
    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f"Thread #{self.pk} by {self.author.username}"

    def can_view(self, user: User) -> bool:
        if self.privacy == Thread.Privacy.PUBLIC:
            return True
        if user.is_authenticated and user == self.author:
            return True
        if self.privacy == Thread.Privacy.FOLLOWERS and user.is_authenticated:
            try:
                author_profile = self.author.profile
                return user in author_profile.followers.all()
            except Profile.DoesNotExist:
                return False
        return False

    @property
    def like_count(self) -> int:
        return self.likes.count()

    @property
    def comment_count(self) -> int:
        return self.comments.count()


class Comment(models.Model):
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='comments')
    content = models.TextField()
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='replies')
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self) -> str:
        return f"Comment #{self.pk} by {self.author.username}"
