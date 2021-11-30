from django.core.management.base import BaseCommand, CommandError
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = 'changes the authorization key for specified users'

    def add_arguments(self, parser):
        parser.add_argument('username', nargs='+')

    def handle(self, *args, **options):
        for username in options['username']:
            try:
                user =  User.objects.get(username=username)
                token = Token.objects.get(user=user)
            except User.DoesNotExist:
               raise CommandError('User "%s" does not exist' % username)
            except Token.DoesNotExist:
               raise CommandError('User "%s" does not have assigned token' % username)

            token.delete()
            Token.objects.create(user=user)
            token = Token.objects.get(user=user)

            self.stdout.write('changed token for user: %s' % username)
            self.stdout.write('new token is: %s' % token)