import math as m
import operator
import copy
import matplotlib.pyplot as plt
import numpy as np
def is_valid(matrix):
    """
    list<list<int>>->bool
    returns true if the matrix is valid
    
    """
    
    valid_lenght=len(matrix[0])
    
    for row in matrix:
        if len(row)!=valid_lenght:
            return False
    return True

def sum_of_rows(rows):
    """
    (list<list>)->list<int>
    returns the sum of the rows
    
    
    
    """
    
    resulting_row=[0]*len(rows[0])
    for row in rows:
        for i,num in enumerate(row):
            resulting_row[i]+=num
    return resulting_row


def multiply_by_constant(matrix,Co):
    """
    (list,int)->list
    returns the matrix multiplied with the matrix without changing the original matrix.
    """
    resulting_matrix=[]
    for row in matrix:
        resulting_row=[]
        for num in row:
            resulting_row.append(round(num*Co,3))
        resulting_matrix.append(resulting_row)
    return resulting_matrix

def create_zero_matrix(n_row,n_column):
    """
    returns a matrix full of 0s with n_rows rows and n_column columns.
    """
    zero_matrix=[]
    for i in range(n_row):
        new=[0]*n_column
        zero_matrix.append(new)
    return zero_matrix

def get_det(matrix):
    """
    returns the determinant of any square matrix.
    """
    row=len(matrix)
    col=len(matrix[0])
    if row!=col:
        raise ValueError("This matrix has no determinant.")
    if row==2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    elif len(matrix[0])<=1:
        return matrix[0][0]
    det=0
    for j in range(col):
        new_one=copy.deepcopy(matrix)
        eliminate_column_and_row(new_one,0,j)
        det+=matrix[0][j]*get_det(new_one)*(-1)**j
    return round(det,3)
          
def eliminate_column_and_row(matrix,index_row,index_column):
    """
    eliminates the designed row and column of the matrix
    """
    matrix.pop(index_row)
    for row in matrix:
        row.pop(index_column)
        
def get_cofactor(matrix):
    """
    returns the cofactor matrix of the matrix without changing the original matrix.
    """
    r=len(matrix)
    c=len(matrix[0])
    cofactor=create_zero_matrix(r,c)
    for i in range(len(cofactor)):
        for j in range(len(cofactor[i])):
            new_one=copy.deepcopy(matrix)
            eliminate_column_and_row(new_one,i,j)
            cofactor[i][j]=round(get_det(new_one)*(-1)**(i+j),3)
    return cofactor

def transpose(matrix):
    """
    returns the transpose matrix of the matrix without changing the original matrix.
    """
    r=len(matrix)
    c=len(matrix[0])
    transposem=create_zero_matrix(c,r)
    for a in range(r):
        for b in range(c):
            transposem[b][a]=matrix[a][b]
    return transposem
       
def adjoint(matrix):
    """
    returns the adjoint matrix of the matrix without changing the original matrix.
    """
    return transpose(get_cofactor(matrix))

def inverse(matrix):
    """
    returns the inverse matrix of the matrix without changing the original matrix.
    """
    det=get_det(matrix)
    if det==0 or len(matrix)!=len(matrix[0]):
        raise ValueError("This matrix is not invertible.")
    return multiply_by_constant(adjoint(matrix),1/det)


class Matrix:
    """
    creates new class Matrix
    instance methods:
    *list: matrix
    

    """
    def __init__(self,matrix):
        """
    (list<list<int>>)->None
        creates new object of type matrix
        
        
        """
        self.matrix=matrix
        if not is_valid(self.matrix):
            raise ValueError("This matrix is not valid, try again.")
        
        self.n_columns=len(matrix[0])
        self.n_rows=len(matrix)
        if self.n_columns==self.n_rows :
            self.det=round(get_det(matrix),3)      
            self.cofactor=get_cofactor(matrix)
            self.transpose=transpose(matrix)
            self.adjoint=adjoint(matrix)
            self.inverse=inverse(matrix)
    
    def __str__(self):
        char='['
        string=''
        for row in self.matrix:
            string+=char+str(row) + '\n'
            char=' '
        return string[:-1]+']'
    
def is_invertible(matrix):
    """
    list<list<int>>->bool
    returns true if the matrix is invertible and vice-versa
    
    """
    return get_det(matrix)!=0
    
        
    
def __eq__(a,b):
    try:
        if len(a)!=len(b) or len(a[0])!=len(b[0]):
            raise ValueError("These matrices are not comparable.")
        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j]!=b[i][j]:
                    return False
        return True
    except TypeError:
        print("These matrices are not comparable.")
        
           
def sum_or_sub(m1,m2,operation):
    """
    (list<list<int>>,list<list<int>>)->list<list<int>>
    returns the sum of the matrices, raises a value error if the sum cant be done.    
    """
    ops={'+':operator.add,'-':operator.sub}
    op_function=ops[operation]
    
    resulting_matrix=[] 
    if len(m2)!=len(m1) or len(m1[0])!=len(m2[0]):
        raise ValueError("These matrices are not compatible for a sum/substraction.")
    for j in range(len(m1)):
        result_row=[]
        for i in range(len(m1[0])):
            result_row.append(op_function(m1[j][i],m2[j][i]))
        resulting_matrix.append(result_row)
    return resulting_matrix
    
    
def summ(m2,m1):
    return sum_or_sub(m2,m1,'+')
    

def sub(m1,m2):
    return sum_or_sub(m1,m2,'-')
    
def matrix_mult(x,y):
    #make sure the shapes match so we can perform matrix multiplication
    assert len(x[0]) == len(y)
    
    #create the result with the correct shape
    result = [[0 for j in range(len(y[0]))] for i in range(len(x))]
    
    #TODO: multiply x and y
    #hint: use nested loops
    for i in range(len(x)):
        for j in range(len(y[0])):
            for n in range(len(y)):
                result[i][j]+=x[i][n]*y[n][j]
            result[i][j]=round(result[i][j],3)

    return result

def vector_to_matrix(real_vector):
    false_vector=create_zero_matrix(len(real_vector),1)
    for i in range(len(real_vector)):
        false_vector[i][0]=real_vector[i]
    return false_vector
    
    
    
    
def matrix_to_vector(false_vector):
    real_vector=[]
    for row in false_vector:
        real_vector.append(row[0])
    return real_vector

def solve_system(matrix,vector):
    
    if not is_invertible(matrix):
        raise ValueError("This system has no solution.")
    elif len(matrix)!=len(matrix[0]):
        raise AssertionError("This system is too complicated for me to solve :(")
    elif len(vector)!=len(matrix):
        raise AssertionError("This vector and matrix are not compatible :(")
    false_vector=matrix_mult(inverse(matrix),vector_to_matrix(vector))
    return matrix_to_vector(false_vector)


def best_fit_line(matrix,images_vector):
    
    return solve_system(matrix_mult(transpose(matrix),matrix),matrix_to_vector(matrix_mult(transpose(matrix),vector_to_matrix(images_vector))))

def create_matrix():
    try:
        n_r=int(input("How many rows?"))
        n_c=int(input("How many columns?"))
        matrix=create_zero_matrix(n_r,n_c)
        for i in range(n_r):
            print("Column number",i+1)
            for j in range(n_c):
                num=float(input("Enter a number"))
                matrix[i][j]=num
        return matrix
    except ValueError:
        print("Please give the numbers in the correct form.")
        create_matrix()
        
def create_vector():
    try:
        n=int(input("How many coordiantes?"))
        vector=[]
        for i in range(n):
            num=float(input("Enter a number"))
            vector.append(num)
        return vector
    except ValueError:
        print("Please give the numbers in the correct form.")
        create_vector()
        
def dot_product(v1,v2):
    dot=0
    for i in range(len(v1)):
        dot+=v1[i]*v2[i]
    return dot
    
    
    
def matrix_solver():
    
    command=input("Hi there, welcome to my matrice solver where you can do every matrice operation you can imagine and more! For starters choose one of these options(enter the number associated to each option):\n 1-Matrix and vector operations\n 2-Solve a linear system\n 3-Best fit line Calculator\n")
    if command=='1' or command.lower()=='Matrix and vector operations':
        command=int(input("Here are the possible operations:\n 1-Single Matrix operations\n 2-Matrix-Matrix operations\n 3-Matrix-Vector operations\n 4-Vector-Vector operations\n"))
        if command==1:
            print("The matrix:\n")
            matrix=create_matrix()
            command=input("Here are the possible operations:\n 1-Check if matrix is invertible \n 2-Get Determinant.\n 3-Get Inverse\n 4-Get Adjoint Matrix 5-Get Cofactor Matrix\n 6-Get Transpose Matrix 7- Multiply by constant.")
            if command=='1':
                return is_invertible(matrix)
            elif command=='2':
                print(get_det(matrix))
            elif command=='3':
                print(inverse(matrix))
            elif command=='4':
                print(adjoint(matrix))
            elif command=='5':
                print(get_cofactor(matrix))
            elif command=='6':
                print(transpose(matrix))
            elif command=='7':
                C=float(input('Enter the constant.'))
                print(multiply_by_constant(matrix,C))
                
        elif command==2:
            print("First Matrix:\n")
            a=create_matrix()
            print("Second Matrix:\n")
            b=create_matrix()
            command=input("Here are the possible operations:\n 1-Check if the matrices are equal \n 2-Sumation.\n 3-Substraction(same order you gave them)\n 4-Product(same order you gave them)\n ")
            if command=='1':
                return __eq__(a,b)
            elif command=='2':
                print(summ(a,b))
            elif command=='3':
                print(sub(a,b))
            elif command=='4':
                print(matrix_mult(a,b))
                
        elif command==3:
            print("The matrix:\n")
            a=create_matrix()
            print("The vector:\n")
            x=create_vector()
            command=input("Here are the possible operations:\n 1-Get the Image of the vector by the matrix.\n 2-Get the coordinates of the vector by the matrix.\n ")
            if command=='1':
                print(matrix_mult(a,vector_to_matrix(x)))
            elif command=='2':
                print(solve_system(a,x))
                
        elif command==4:
            print("First Vector:\n")
            a=create_vector()
            print("Second Vector:\n")
            b=create_vector()
            if len(a)!=len(b):
                raise AssertionError("Sorry you can only do operations on vectors with the same number of coordinates.")
            command=input("Here are the possible operations:\n 1-Get dot product.\n 2-Get cross product.\n ")
            if command=='1':
                print(dot_product(a,b))
            elif command=='2':
                print(cross_product(a,b))
    elif  command=='2':
        n_e=int(input("How many equations are there?	"))
        n_v=int(input("How many variables are there?	"))
        matrix=create_zero_matrix(n_e,n_v)
        vector=[]
        equations=''
        for i in range(n_e):
            print('Equation number',i+1)
            equation=''
            for j in range(n_v):
                number=float(input("Enter the factor for the variable number "+str(j+1)+'\n'))
                matrix[i][j]=number
                equation+=str(number)+'*x'+str(1+j)+'+'
            c=float(input("Enter what it's equal to:"))
            vector.append(c)
            equations+=equation[:-1]+'='+str(c)+'\n'
        solution=solve_system(matrix,vector)
        sol=''
        for i in range(len(solution)):
            sol+='x'+str(i+1)+'='+str(solution[i])+'\n'
        print('The solution to the system:\n'+equations+'is\n'+sol)
    elif  command=='3':
        n_c=int(input('What is the highest degree for the function?\n'))+1
        n_r=int(input('How many points ?\n'))
        vector=[]
        x_coordiantes=[]
        y_coordiantes=[]
        matrix=create_zero_matrix(n_r,n_c)
        for i in range(n_r):
            print('Point'+str(i+1)+'\n')
            x=float(input('Give the x-coordinate of the point: '))
            x_coordiantes.append(x)
            y=float(input('Give the y-coordinate of the point: '))
            y_coordiantes.append(y)
            for j in range(n_c):
                matrix[i][j]=x**(j)
            vector.append(y)
        solutions=best_fit_line(matrix,vector)
        max=largest_value(x_coordiantes)
        x = np.linspace(-3*max,3*max, 100)
        y_sol=0
        equation='y='
        for i in range(len(solutions)-1,-1,-1):
            y_sol+=x**i*solutions[i]
            if i>1:
                equation+=str(solutions[i])+'*x^'+str(i)+'+'
            elif i==1:
                equation+=str(solutions[i])+'*x'+'+'
            elif i==0:
                equation+=str(solutions[i])+'+'
        # use set_position
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
 
        # depict illustration
        plt.xlim(-2*max,2*max)
        plt.plot(x, y_sol)
        plt.grid(True)
        print('The equation of the best fit line is: \n'+ equation[:-1])
        plt.show()
def largest_value(lis):
    value=lis[0]
    for i in range(len(lis)):
        if value<lis[i]:
            value=lis[i]
    return value

                

matrix_solver()

    

print('Thanks for using my Matrice solver! Have a nice day.(RUN THE SYSTEM AGAIN TO USE MY MATRICE SOLVER.)')


            


    

