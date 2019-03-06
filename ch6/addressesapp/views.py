from django.shortcuts import render, redirect, render_to_response
from django.template  import RequestContext, loader
from django.urls import reverse
from django.conf import settings
import logging, os, json, datetime, urllib
from .models import Person
from ast import literal_eval


# 작가별 index 페이지            
def addressesbook(request):
    logging.debug('address book')
    get_data, context = request.GET, {}
    letter            = get_data.get('letter',None)
    if letter:
        contacts = Person.objects.filter(name__iregex=r"(^|\s)%s" % letter)
    else:
        contacts = Person.objects.all()
    # Sorted Alphabetically
    contacts            = sort_lower(contacts,"name")
    context['contacts'] = contacts
    alphabet_string     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    context['alphabet'] = [l for l in alphabet_string]
    return render(request, 'addressesapp/book.html', context) 


# 객체 내 정렬함수
def sort_lower(list_, key_name):
    # key = str.lower : 정렬기준일 뿐, 실제로 객체를 변환하지 않는다
    # getattr() 클래스내 객체, 딕셔너리 키값 내부 객체
    return sorted(list_, key = lambda item: getattr(item, key_name).lower())


# 작가별 index 페이지 'delete' 함수
def delete_person(request,name):
    if Person.objects.filter(name=name).exists():
       p = Person.objects.get(name=name)
       p.delete()
       
    # GET & Sorted Alphabetically
    contacts, context   = Person.objects.all(), {}
    contacts            = sort_lower(contacts,"name") 
    context['contacts'] = contacts  # contacts.order_by("name")
    return render(request, 'addressesapp/book.html', context)


# 원하는 결과가 없는경우
def notfound(request):
    return render(request, 'addressesapp/nopersonfound.html')


# 
def get_contacts(request):
    logging.debug('here')
    if request.method == 'GET':
        get_data = request.GET
        data     = get_data.get('term','')
        print ('get contacts:', data)
        if data == '':
            return render(request, 'addressesapp/nopersonfound.html')
        else:
            return redirect('%s?%s' % (reverse('addressesapp.views.addressesbook'),
                                urllib.urlencode({'letter': data})))


def main(request):    
    context = {}
    if request.method == 'POST':
        post_data, data = request.POST, {}
        data['name']    = post_data.get('name', None)
        data['email']   = post_data.get('email', None)
        if data:
            return redirect('%s?%s' % (reverse('home'), #addressesapp.views.main'),
                                urllib.request(data={'q': data})))

    elif request.method == 'GET':
        get_data = request.GET 
        data     = get_data.get('q', None)
        if not data:
            return render(request, 'addressesapp/home.html', context)
        data = literal_eval(get_data.get('q', None))
        print (data)
        if not data['name'] and not data['email']:
            return render(request, 'addressesapp/home.html', context)
                
        # add person to emails address book or update
        if Person.objects.filter(name = data['name']).exists():
            p      = Person.objects.get(name = data['name'])
            p.mail = data['email']
            p.save()
        else:
            p      = Person()
            p.name = data['name']
            p.mail = data['email']
            p.save()
            
        # restart page
        return render(request, 'addressesapp/home.html', context)