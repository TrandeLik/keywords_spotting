|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Мера сходства элементов штрихового представления рукописного текста на основе Фурье-дескриптора
    :Тип научной работы: M1P
    :Автор: Дмитрий Дмитриевич Феоктистов
    :Научный руководитель: д.т.н., Местецкий Леонид Моисеевич

Аннотация
========

В работе изучается задача классификации элементов штрихового разложения рукописного текста. Для ее решения предлагается использовать векторизацию штрихов на основе преобразования Фурье, после чего использовать полученные вектора в метрических классификаторах. Полученный в ходе работы классификатор штрихов предлагается использовать как часть ранжирующего алгоритма для задачи поиска ключевых слов в рукописном контексте. Решение этой задачи может значительно упростить работу с архивными данными. В ходе экспериментов с изображениями панграмм показано, что для описания штриха достаточно малого количества коэффициентов (5-7 штук), а также, что предлагаемый подход является наилучшим по соотношению скорость/качество 

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. D. Feoktistov, L. Mestetskiy ''A similarity measure of elements of the stroke representation
of handwritten text based on Fourier descriptor'', MMPR-21

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
