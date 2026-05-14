from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Request


class Command(BaseCommand):
    help = "Seed database with demo data for Sangel."

    def handle(self, *args, **options):
        User = get_user_model()
        with transaction.atomic():
            Request.objects.all().delete()
            User.objects.all().delete()

            operator = User.objects.create_superuser(
                username="operator@sangel.ru",
                email="operator@sangel.ru",
                password="123",
            )
            operator.role = User.ROLE_OPERATOR
            operator.is_staff = True
            operator.is_superuser = True
            operator.save(update_fields=["role", "is_staff", "is_superuser"])

            clients = []
            for idx in range(1, 4):
                email = f"client{idx}@sangel.ru"
                client = User.objects.create_user(
                    username=email, email=email, password="123", role=User.ROLE_CLIENT
                )
                clients.append(client)

            alpha = User.objects.create_user(
                username="alpha@sangel.ru",
                email="alpha@sangel.ru",
                password="123",
                role=User.ROLE_TEAM,
                team_name="Альфа",
            )
            beta = User.objects.create_user(
                username="beta@sangel.ru",
                email="beta@sangel.ru",
                password="123",
                role=User.ROLE_TEAM,
                team_name="Бета",
            )

            Request.objects.create(
                client=clients[0],
                address="ул. Ленина, 10",
                description="Тревожная кнопка сработала в магазине.",
                status=Request.STATUS_NEW,
            )
            Request.objects.create(
                client=clients[1],
                address="пр. Мира, 5",
                description="Нужна группа реагирования на склад.",
                status=Request.STATUS_ACCEPTED,
                assigned_team=alpha,
            )
            Request.objects.create(
                client=clients[2],
                address="ул. Советская, 21",
                description="Подозрительные лица у входа.",
                status=Request.STATUS_EN_ROUTE,
                assigned_team=beta,
            )
            Request.objects.create(
                client=clients[0],
                address="ул. Гагарина, 7",
                description="Ложный вызов, ситуация разрешена.",
                status=Request.STATUS_COMPLETED,
                assigned_team=alpha,
            )
            Request.objects.create(
                client=clients[1],
                address="ул. Молодежная, 3",
                description="Клиент отменил вызов.",
                status=Request.STATUS_CANCELLED,
            )

        self.stdout.write(self.style.SUCCESS("Демо-данные успешно созданы."))
