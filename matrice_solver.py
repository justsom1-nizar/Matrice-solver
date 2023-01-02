import math as m
import numpy as np
import operator
import copy
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
    r=len(matrix)
    c=len(matrix[0])
    if r!=c:
        raise ValueError("This matrix has no determinant.")
    if r==2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    det=0
    for j in range(c):
        new_one=copy.deepcopy(matrix)
        eliminate_column_and_row(new_one,0,j)
        det+=matrix[0][j]*get_det(new_one)*(-1)**j
    return det
          
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
    transpose=create_zero_matrix(r,c)
    for a in range(len(matrix)):
        for b in range(len(matrix[a])):
            transpose[b][a]=matrix[a][b]
    return transpose
       
def adjoint(matrix):
    """
    returns the adjoint matrix of the matrix without changing the original matrix.
    """
    return transpose(get_cofactor(matrix))

def inverse(matrix):
    """
    returns the inverse matrix of the matrix without changing the original matrix.
    """
    det=np.linalg.det(matrix)
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
    
    def is_invertible(self):
        """
        list<list<int>>->bool
        returns true if the matrix is invertible and vice-versa
        >>>
        """
        return self.n_columns==self.n_rows and self.det!=0
    
    def multiply_by_constant(self,Con):
        self.matrix=multiply_by_constant(self.matrix,Con)
        
    
    def __eq__(self,m):
        if self.n_rows!=self.n_columns or m.n_rows!=m.n_columns:
            raise ValueError("These matrices are not comparable.")
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                if self.matrix[i][j]!=m.matrix[i][j]:
                    return False
        return True
    
    @staticmethod        
    def sum_or_sub(m1,m2,operation):
        """
        (list<list<int>>,list<list<int>>)->list<list<int>>
        returns the sum of the matrices, raises a value error if the sum cant be done.    
        """
        ops={'+':operator.add,'-':operator.sub}
        op_function=ops[operation]
    
        resulting_matrix=[] 
        if m1.n_rows!=m2.n_rows or m1.n_columns!=m2.n_columns:
            raise ValueError("These matrices are not compatible for a sum/substraction.")
        for j in range(m1.n_rows):
            result_row=[]
            for i in range(m1.n_columns):
                result_row.append(op_function(m1.matrix[j][i],m2.matrix[j][i]))
            resulting_matrix.append(result_row)
        return resulting_matrix
    
    @staticmethod
    def summ(m2,m1):
        return Matrix.sum_or_sub(m2,m1,'+')
    
    @staticmethod
    def sub(m1,m2):
        return Matrix.sum_or_sub(m1,m2,'-')
    
    @staticmethod
    def product(m1,m2):
        if m1.n_columns!=m2.n_rows:
            raise ValueError("These matrices are not compatible for a multplication in this order.")
        resulting_matrix=create_zero_matrix(m1.n_rows,m2.n_columns)
        for i in range(m1.n_rows):
            for j in range(m2.n_columns):
                for n in range(m2.n_rows):
                    resulting_matrix[i][j]+=m1.matrix[i][n]*m2.matrix[n][j]
                
        return resulting_matrix
a=Matrix([[1,2,4],[2,5,6]])
b=Matrix([[1,8],[6,4],[2,4]])
print(Matrix.product(a,b))


            


    

