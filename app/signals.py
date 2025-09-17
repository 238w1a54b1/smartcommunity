# app/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Profile
import requests
from django.core.files.base import ContentFile
from allauth.account.signals import user_signed_up

# 1. Create Profile when a User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

# 2. Save Profile when User is saved
@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()

# 3. Fetch social profile picture for Google/Github login
@receiver(user_signed_up)
def populate_profile_social_login(request, user, **kwargs):
    sociallogin = kwargs.get('sociallogin')
    if sociallogin:
        provider = sociallogin.account.provider
        profile = Profile.objects.get(user=user)
        
        # Fetch profile picture from provider
        if provider == 'google':
            picture_url = sociallogin.account.extra_data.get('picture')
            user.first_name = sociallogin.account.extra_data.get('given_name', '')
            user.last_name = sociallogin.account.extra_data.get('family_name', '')
            user.save()
        elif provider == 'github':
            picture_url = sociallogin.account.extra_data.get('avatar_url')
            user.first_name = sociallogin.account.extra_data.get('name', '').split(' ')[0]
            user.last_name = ' '.join(sociallogin.account.extra_data.get('name', '').split(' ')[1:])
            user.save()
        else:
            picture_url = None

        # Save profile image
        if picture_url:
            response = requests.get(picture_url)
            if response.status_code == 200:
                profile.profile_image.save(
                    f'{user.username}_{provider}.jpg',
                    ContentFile(response.content),
                    save=True
                )
