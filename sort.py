import time
# import math
# import matplotlib.patches as mpatches
import random
import matplotlib.pyplot as plt
# import sys
# 1sys.setrecursionlimit(10000)


'''
//////////////////////////////////////////////////////////////////////
/// Klasa Sort zawiera procedury sortowania przez scalanie, Shella,
/// przez kopcowanie, szybkie. Inicjalizujemy klasę metodą init (a'la konstruktor)  
/// pierwszym argumentem w kazdej metodzie jest self jest referencja do bieżacej instancji klasy. 
/// W klasie przechowujemy tablice danych do posortowania i czasy działania algorytmów sortowania.
/// Metody do_sort sortuja tablice i rysuja wykresy zaleznosci czasow od rozmiaru tablicy
//////////////////////////////////////////////////////////////////////
'''


class Sort:
    def __init__(self, Array):

        self.Array = Array
        self.data_size = len(Array)

        self.merge_sort_time = 0
        self.quick_sort_time = 0
        self.shell_sort_time = 0
        self.heap_sort_time = 0
#       self.intro_sort_time = 0

    """  Sortowanie przez scalanie  """

    def do_merge_sort(self):
        # Kopiujemy dane
        merge_sort_arr = list(self.Array)

        # Czas sortowania przez scalanie
        start = time.time()
        self.merge_sort(merge_sort_arr)
        end = time.time()

        # Rysujemy wykres punktowy czas dzialania od rozmiaru tablicy
        self.merge_sort_time = (end - start)
        print("Czas sortowania przez scalanie: ", str(self.merge_sort_time))
        plt.scatter(len(merge_sort_arr), self.merge_sort_time, color="b")

    """  Sortowanie metodą Shella  """

    def do_shell_sort(self):
        shell_sort_arr = list(self.Array)

        start = time.time()
        self.shell_sort(shell_sort_arr)
        end = time.time()

        self.shell_sort_time = (end - start)
        print("Czas dla sortowania Shella: ", str(self.shell_sort_time))
        plt.scatter(len(shell_sort_arr), self.shell_sort_time, color="c")

    """  Sortowanie przez kopcowanie  """

    def do_heap_sort(self):
        heap_sort_arr = list(self.Array)

        start = time.time()
        self.heap_sort(heap_sort_arr)
        end = time.time()

        self.heap_sort_time = (end - start)
        print("Czas sortowania przez kopcowanie: ", str(self.heap_sort_time))
        plt.scatter(len(heap_sort_arr), self.heap_sort_time, color="r")

    """  Sortowanie szybkie  """

    def do_quick_sort(self):
        quick_sort_arr = list(self.Array)

        start2 = time.time()
        self.quick_sort(quick_sort_arr, 0, len(quick_sort_arr)-1)
        end2 = time.time()

        self.quick_sort_time = (end2 - start2)
        print("Czas dla sortowania szybkiego: ", str(self.quick_sort_time))
        plt.scatter(len(quick_sort_arr), self.quick_sort_time, color="g")

    """ Sortowanie przez wstawianie, introspektywne
    
    def do_insertion_sort(self):
        insertion_sort_arr = list(self.Array)
       
        start2 = time.time()
        self.insertion_sort(insertion_sort_arr)
        end2 = time.time()      
        print(insertion_sort_arr)
        self.insertion_sort_time = (end2 - start2)
        print("Insertion sort time: ", str(self.insertion_sort_time))
        plt.scatter(len(insertion_sort_arr), self.insertion_sort_time, color="r")
    

    def do_intro_sort(self):
        intro_sort_arr = list(self.Array)

        start = time.time()
        self.intro_sort(intro_sort_arr,0 , len(intro_sort_arr)-1)
        end = time.time()
        # Plot the time taken and the data size
        self.intro_sort_time = (end - start)
        print("Shell sort time: ", str(self.intro_sort_time))
        plt.scatter(len(intro_sort_arr), self.intro_sort_time, color="r")
    
    
    def insertion_sort(self, arr):
        for i in range(1, len(arr)):
            dec = i - 1
            elem = arr[i]
            while dec >= 0 and arr[dec] > elem:
                arr[dec + 1] = arr[dec]
                dec = dec - 1
            arr[dec + 1] = elem

    """

    def merge_sort(self, arr):
        # Base case
        if len(arr) > 1:  # wiecej niz jeden element
            middle = int(len(arr) / 2)  # wyznaczamy srodek

            # Dzielimy na dwie podtablice left i right
            left = arr[:middle]
            right = arr[middle:]

            # Sorujemy dwie podtablice, rekurencyjnie wywolujemy metode merge_sort
            self.merge_sort(left)
            self.merge_sort(right)

            # Indeksy pomocnicze początek pierwszej i drugiej podtablicy, indeks obecny pobieramy do tablicy wynikowej
            left_index = 0
            right_index = 0
            current = 0

            # Dopoki nie wyczerpia sie elementy z podtablic
            while left_index < len(left) and right_index < len(right):
                if left[left_index] <= right[right_index]:
                    arr[current] = left[left_index]
                    left_index += 1

                else:
                    arr[current] = right[right_index]
                    right_index += 1
                current += 1
# Sprawdzamy czy zostaly elementy w podtablicach gdy petla przerwala dzialanie gdy elementy podtablicy sie wykorzystaly
            while left_index < len(left):
                arr[current] = left[left_index]
                current += 1
                left_index += 1

            while right_index < len(right):
                arr[current] = right[right_index]
                current += 1
                right_index += 1

    def shell_sort(self, arr):
        n = len(arr)
        dist = n//2
        while dist > 0:
            for i in range(dist, n):
                temp = arr[i]
                j = i
                while j >= dist and arr[j-dist] > temp:
                    arr[j] = arr[j-dist]
                    j -= dist
                arr[j] = temp
            dist //= 2

    def heapify(self, arr, heap_size, i):
        largest = i  # maksymalny indeks, indeks rodzica
        left = 2 * i + 1  # lewwe dziecko
        right = 2 * i + 2  # prawe dziecko

        # sprawdzamy czy miesci sie w tablicy i czy lewe dziecko jest wieksze od rodzica
        if left < heap_size and arr[i] < arr[left]:
            largest = left

        # sprawdzamy czy miesci sie w tablicy i czy prawe dziecko jest wieksze od rodzica
        if right < heap_size and arr[largest] < arr[right]:
            largest = right

        if largest != i:  # sprawdzamy czy maksymalny indeks zmienil swoja wartosc
            arr[i], arr[largest] = arr[largest], arr[i]  # zmiana rodzica
            # sprawdzamy czy po zmianie rodzic jest w odpowiedniej pozycji
            self.heapify(arr, heap_size, largest)

    def heap_sort(self, arr):
        heap_size = len(arr)

        # Budyjemy maksymalny kopiec
        for i in range(heap_size, -1, -1):
            self.heapify(arr, heap_size, i)
        # zamieniamy korzen z ostatnim elementem zmniejszamy kopiec o 1 i ponownie wywolujemy
        for i in range(heap_size - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self.heapify(arr, i, 0)

    def partition(self, arr, left, right):      # procedura partycjonowania która dzieli tablice na dwa podzbiory
        border = (left - 1)         # granica ktory zwiekszmy gdy dokonujemy zamiany, pomaga nam przejsc przez tablice
        pivot = arr[right]          # za tzw. element osiowy wybieramy ostatni element tablicy

        for j in range(left, right):            # j = licznik pozwala przemieszczac sie od lewej do prawej w tablicy

            # Jesli element jest mniejszy lub rowny od pivota to przesuwamy granice o 1 i zamieniamy
            if arr[j] <= pivot:
                border = border + 1
                arr[border], arr[j] = arr[j], arr[border]

        arr[border + 1], arr[right] = arr[right], arr[border + 1]
        return border + 1

        # Główna funkcja do implementacji quicksorta
        # arr --> tavlica do posortowania
        # left  --> pierwszy element tablicy
        # right --> ostatni element tablicy

    def quick_sort(self, arr, left, right):
        if left < right:

            # ustawiamy pivot na wlasciwym miejscu
            new_pivot = self.partition(arr, left, right)

            # partition and after partition
            self.quick_sort(arr, left, new_pivot - 1)   # wywolujemy quicksort od poczatku do pivota (1 podproblem)
            self.quick_sort(arr, new_pivot + 1, right)  # wywolujemy od pivota do prawego indeksu(2 podproblem)

    """
    def MedianOfThree(self, a, b, d):
        arr = []
        A = arr[a]
        B = arr[b]
        C = arr[d]

        if A <= B and B <= C:
            return b
        if C <= B and B <= A:
            return b
        if B <= A and A <= C:
            return a
        if C <= A and A <= B:
            return a
        if B <= C and C <= A:
            return d
        if A <= C and C <= B:
            return d

            # The main function that implements Introsort

    # low  --> Starting index,
    # high  --> Ending index
    # depthLimit --> recursion level

    def IntrosortUtil(self, arr, begin, end, depthLimit):
        
        size = end - begin
        if size < 16:
            # if the data set is small, call insertion sort

            self.insertion_sort(arr)
            return

        if depthLimit == 0:
            # if the recursion limit is occurred call heap sort

            self.heap_sort(arr)
            return

        pivot = self.MedianOfThree(begin, begin + size // 2, end)
        (arr[pivot], arr[end]) = (arr[end], arr[pivot])

        # partitionPoint is partitioning index,
        # arr[partitionPoint] is now at right place

        partitionPoint = self.partition(arr, begin, end)

        # Separately sort elements before partition and after partition

        self.IntrosortUtil(arr, begin, partitionPoint - 1, depthLimit - 1)
        self.IntrosortUtil(arr, partitionPoint + 1, end, depthLimit - 1)

        # A utility function to begin the Introsort module

    def intro_sort(self, arr, begin, end):

        # initialise the depthLimit as 2 * log(length(data))

        depthLimit = 2 * math.log2(end - begin)
        self.IntrosortUtil(arr, begin, end, depthLimit)
    """
# Tablice do posortowania: losowe wartosci, 25% poczatkowych posortowanych, 50% posortowanych itd.


def rand(iterations, lower_bound, upper_bound):
    arr = []
    for j in range(iterations):
        arr.append(random.randint(lower_bound, upper_bound))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def percentage_of_sorted_25(iterations, lower_bound, upper_bound):
    arr = []
    percentage = 250
    x = iterations * percentage/1000
    for j in range(iterations):
        if j < x:
            arr.append(j)
        else:
            arr.append(random.randint(lower_bound, upper_bound)+int(x))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def percentage_of_sorted_50(iterations, lower_bound, upper_bound):
    arr = []
    percentage = 500
    x = iterations * percentage/1000
    for j in range(iterations):
        if j < x:
            arr.append(j)
        else:
            arr.append(random.randint(lower_bound, upper_bound)+int(x))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def percentage_of_sorted_75(iterations, lower_bound, upper_bound):
    arr = []
    percentage = 750
    x = iterations * percentage/1000
    for j in range(iterations):
        if j < x:
            arr.append(j)
        else:
            arr.append(random.randint(lower_bound, upper_bound)+int(x))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def percentage_of_sorted_95(iterations, lower_bound, upper_bound):
    arr = []
    percentage = 950
    x = iterations * percentage/1000
    for j in range(iterations):
        if j < x:
            arr.append(j)
        else:
            arr.append(random.randint(lower_bound, upper_bound)+int(x))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def percentage_of_sorted_99(iterations, lower_bound, upper_bound):
    arr = []
    percentage = 990
    x = iterations * percentage/1000
    for j in range(iterations):
        if j < x:
            arr.append(j)
        else:
            arr.append(random.randint(lower_bound, upper_bound)+int(x))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def percentage_of_sorted_997(iterations, lower_bound, upper_bound):
    arr = []
    percentage = 997
    x = iterations * percentage/1000
    for j in range(iterations):
        if j < x:
            arr.append(j)
        else:
            arr.append(random.randint(lower_bound, upper_bound)+int(x))
    print("Rozmiar tablicy: ", len(arr))
    return arr


def sorted_reverse(iterations):
    arr = []
    for j in range(iterations):
        arr.append(j)
    arr.reverse()
    print("Rozmiar tablicy: ", len(arr))
    return arr


INCREMENTER = 100000
DOLNA_GRANICA = 0
GORNA_GRANICA = 1000000

# Wybieramy tablice

for k in range(DOLNA_GRANICA, GORNA_GRANICA, INCREMENTER):

    Arr = rand(k, 0, 1000000)
#    Arr = percentage_of_sorted_25(k, 0, 1000000)
#    Arr = percentage_of_sorted_50(k, 0, 1000000)
#    Arr = percentage_of_sorted_75(k, 0, 1000000)
#    Arr = percentage_of_sorted_95(k, 0, 1000000)
#    Arr = percentage_of_sorted_99(k, 0, 1000000)
#    Arr = percentage_of_sorted_997(k, 0, 1000000)
#    Arr = sorted_reverse(k)
    SortObject = Sort(Arr)  # inicjujjemy obiekt
    # print(Arr)
    print()
    print()

    # Wywołujemy sortowania
    SortObject.do_merge_sort()
    SortObject.do_shell_sort()
    SortObject.do_heap_sort()
#    SortObject.do_quick_sort()


plt.ylim([0, 18])
plt.ylabel('Czas działania [s]')
plt.xlabel('Rozmiar Tablicy')
plt.title('Czasy dla losowych danych')
plt.legend(('Sortowanie przez scalanie', 'Sortowanie Shella', 'Sortowanie przez kopcowanie', 'Sortowanie szybkie'))
plt.show()
