from django.core.management.base import BaseCommand, CommandError
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = 'gets token associated with given users'

    def add_arguments(self, parser):
        parser.add_argument('username', nargs='+')

    def handle(self, *args, **options):
            for username in options['username']:
                try:
                    user = User.objects.get(username=username)
                    token = Token.objects.get(user=user)
                except User.DoesNotExist:
                   raise CommandError('User "%s" does not exist' % username)
                except Token.DoesNotExist:
                   raise CommandError('User "%s" does not have assigned token' % username)

                self.stdout.write('%s: %s' % (username, token))