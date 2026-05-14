from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import Request, User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    fieldsets = BaseUserAdmin.fieldsets + (
        ("Дополнительно", {"fields": ("role", "team_name", "phone_number")}),
    )
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ("Дополнительно", {"fields": ("role", "team_name", "phone_number")}),
    )
    list_display = ("id", "username", "email", "role", "team_name", "is_staff")
    list_filter = ("role", "is_staff", "is_superuser", "is_active")
    search_fields = ("email", "username", "team_name")


@admin.register(Request)
class RequestAdmin(admin.ModelAdmin):
    list_display = ("id", "client", "assigned_team", "status", "created_at")
    list_filter = ("status",)
    search_fields = ("client__email", "assigned_team__email", "address")
    autocomplete_fields = ("client", "assigned_team")
