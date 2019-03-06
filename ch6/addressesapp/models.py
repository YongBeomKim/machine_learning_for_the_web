from django.db import models
from django.utils.translation import ugettext_lazy as lazy

# https://djangobook.com/internationalization-python-code/
# lazy('Name') 은 Global Name Space를 활용하는 용도로 사용됩니다.
# 예전에는 gettext() 함수를 사용했었습니다.


class Person(models.Model):
    name = models.CharField( lazy('Name'), max_length=255, unique=True)
    mail = models.EmailField(max_length=255, blank=True)

    # display name on admin panel
    # lazy('Test') 에서 외부로 출력되는 이름 (Table에는 적용X)
    def __str__(self):
        return self.name
            
    class Meta:
        ordering = ['name']
