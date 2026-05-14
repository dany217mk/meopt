from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    ROLE_CLIENT = "client"
    ROLE_TEAM = "team"
    ROLE_OPERATOR = "operator"

    ROLE_CHOICES = [
        (ROLE_CLIENT, "Клиент"),
        (ROLE_TEAM, "Группа реагирования"),
        (ROLE_OPERATOR, "Оператор"),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_CLIENT)
    team_name = models.CharField(max_length=100, blank=True, null=True)
    phone_number = models.CharField(max_length=30, blank=True)
    email = models.EmailField(unique=True)

    def __str__(self):
        if self.role == self.ROLE_TEAM and self.team_name:
            return f"{self.team_name} ({self.email})"
        return self.email or self.username


class Request(models.Model):
    STATUS_NEW = "new"
    STATUS_ACCEPTED = "accepted"
    STATUS_EN_ROUTE = "en_route"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"

    STATUS_CHOICES = [
        (STATUS_NEW, "Новая"),
        (STATUS_ACCEPTED, "Принята"),
        (STATUS_EN_ROUTE, "В пути"),
        (STATUS_COMPLETED, "Выполнена"),
        (STATUS_CANCELLED, "Отменена"),
    ]

    client = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="requests", on_delete=models.CASCADE
    )
    assigned_team = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name="assigned_requests",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        limit_choices_to={"role": User.ROLE_TEAM},
    )
    address = models.CharField(max_length=255)
    description = models.TextField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_NEW)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_active(self):
        return self.status in {self.STATUS_ACCEPTED, self.STATUS_EN_ROUTE}

    def __str__(self):
        return f"Заявка #{self.pk}"
