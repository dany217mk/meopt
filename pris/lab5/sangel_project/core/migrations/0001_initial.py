from django.conf import settings
from django.contrib.auth.models import UserManager
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.db import migrations, models
from django.utils import timezone


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.CreateModel(
            name="User",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("password", models.CharField(max_length=128, verbose_name="password")),
                (
                    "last_login",
                    models.DateTimeField(
                        blank=True, null=True, verbose_name="last login"
                    ),
                ),
                (
                    "is_superuser",
                    models.BooleanField(
                        default=False,
                        help_text=(
                            "Designates that this user has all permissions without "
                            "explicitly assigning them."
                        ),
                        verbose_name="superuser status",
                    ),
                ),
                (
                    "username",
                    models.CharField(
                        max_length=150,
                        unique=True,
                        validators=[UnicodeUsernameValidator()],
                        verbose_name="username",
                        error_messages={
                            "unique": "A user with that username already exists."
                        },
                    ),
                ),
                (
                    "first_name",
                    models.CharField(
                        blank=True, max_length=150, verbose_name="first name"
                    ),
                ),
                (
                    "last_name",
                    models.CharField(
                        blank=True, max_length=150, verbose_name="last name"
                    ),
                ),
                (
                    "email",
                    models.EmailField(
                        max_length=254, unique=True, verbose_name="email address"
                    ),
                ),
                (
                    "is_staff",
                    models.BooleanField(
                        default=False,
                        help_text="Designates whether the user can log into this admin site.",
                        verbose_name="staff status",
                    ),
                ),
                (
                    "is_active",
                    models.BooleanField(
                        default=True,
                        help_text=(
                            "Designates whether this user should be treated as active. "
                            "Unselect this instead of deleting accounts."
                        ),
                        verbose_name="active",
                    ),
                ),
                (
                    "date_joined",
                    models.DateTimeField(default=timezone.now, verbose_name="date joined"),
                ),
                (
                    "role",
                    models.CharField(
                        choices=[
                            ("client", "Клиент"),
                            ("team", "Группа реагирования"),
                            ("operator", "Оператор"),
                        ],
                        default="client",
                        max_length=20,
                    ),
                ),
                ("team_name", models.CharField(blank=True, max_length=100, null=True)),
                ("phone_number", models.CharField(blank=True, max_length=30)),
                (
                    "groups",
                    models.ManyToManyField(
                        blank=True,
                        help_text=(
                            "The groups this user belongs to. A user will get all permissions "
                            "granted to each of their groups."
                        ),
                        related_name="user_set",
                        related_query_name="user",
                        to="auth.group",
                        verbose_name="groups",
                    ),
                ),
                (
                    "user_permissions",
                    models.ManyToManyField(
                        blank=True,
                        help_text="Specific permissions for this user.",
                        related_name="user_set",
                        related_query_name="user",
                        to="auth.permission",
                        verbose_name="user permissions",
                    ),
                ),
            ],
            options={"abstract": False},
            managers=[("objects", UserManager())],
        ),
        migrations.CreateModel(
            name="Request",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("address", models.CharField(max_length=255)),
                ("description", models.TextField()),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("new", "Новая"),
                            ("accepted", "Принята"),
                            ("en_route", "В пути"),
                            ("completed", "Выполнена"),
                            ("cancelled", "Отменена"),
                        ],
                        default="new",
                        max_length=20,
                    ),
                ),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "assigned_team",
                    models.ForeignKey(
                        blank=True,
                        limit_choices_to={"role": "team"},
                        null=True,
                        on_delete=models.SET_NULL,
                        related_name="assigned_requests",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "client",
                    models.ForeignKey(
                        on_delete=models.CASCADE,
                        related_name="requests",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
