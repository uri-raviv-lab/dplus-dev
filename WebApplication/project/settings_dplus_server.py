import os

STATIC_ROOT = '/home/dplus/dplus/web/static'
LOG_DIR = '/home/dplus/dplus/web/logs'
EXE_DIR = "/home/dplus/dplus/src/cmake"

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': '/home/dplus/dplus/web/db.sqlite3',
    },
}

try:
    os.makedirs(LOG_DIR)
except OSError:
    if not os.path.isdir(LOG_DIR):
        raise

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'errors.log'),
        },
        'dplus_file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'dplus.log')
        }
    },
    'loggers': {
        'root': {
            'handlers': ['error_file'],
            'level': 'ERROR',
            'propagate': True,
        },
        'database_items': {
            'handlers': ['dplus_file'],
            'level': 'DEBUG',
            'propagate': True,
        }
    },
}
