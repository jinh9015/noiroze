from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from django.contrib.auth.models import User


class CustomUserAdmin(admin.ModelAdmin) :
    fieldsets = (
        (None, {'fields': ('userid', 'password', 'email', 'name', 'apartment', 'dong', 'ho')}),
        (('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        (('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('userid', 'password1', 'password2', 'email', 'name', 'apartment', 'dong', 'ho'),
        }),
    )
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'groups')
    search_fields = ('userid', 'email', 'dong', 'ho')
    ordering = ('userid', 'email', 'name', 'dong', 'ho')
    list_display = ('userid', 'email', 'name', 'dong', 'ho', 'is_staff')


admin.site.register(CustomUser, CustomUserAdmin)


    