#include <mpi.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>

int N = 7;
int M = 5;
int K = 7;

int procSize;
int procRank;
int rankX;
int rankY;
/*
 *     N - number of rows in A   and C
 *     M - number of columns in A
 *     K - number of columns in B and C
 *
 *        M = 3         K = 2
 *
 *  N   (q, w, e)                  ( z, x)
 *  =  ( r, t, y )     ( a, b )   ( b, n  )
 *  4  ( u, i, o ) *  ( c, d  ) = ( m, k  )
 *      (p, a, s)      (e, f )     (k, l )
 *
 *
 * */

/*
 * inRowDispls is offset of proc for rowComm
 * */
std::vector<int> inRowDispls;
/*
 * inCowDispls is offset of proc for colComm
 * */
std::vector<int> inColDispls;

std::vector<int> inRowSizes;   // cols per proc
std::vector<int> inColSizes;   // rowsOerProc


MPI_Datatype rowType;

MPI_Datatype colType;

double *transpose(double *pDouble, int inColsSize, int inRowsSize);

void initMatrixes(double** matrixA, double** matrixB, double** matrixC){
    *matrixA = new double[N * M];
    *matrixB = new double[M * K];
    *matrixC = new double[N * K];

    for( int i = 0; i < N * M; ++i ){
        (*matrixA)[i] = i;//std::rand();
    }
    for( int i = 0; i < M* K; ++i ){
        (*matrixB)[i] = i;//std::rand();
    }

    std::fill_n(*matrixC, N* K, 0);
}
void makeCommEnvironment(MPI_Comm *comm2d, MPI_Comm *dimComms, int *procsInDirection){
    int periods[2] = {0, 0};
    int remain[2] = {1, 0};

    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procSize);
    MPI_Dims_create(procSize, 2, procsInDirection);
    //std::swap(procsInDirection[0],procsInDirection[1]);

    MPI_Cart_create(MPI_COMM_WORLD, 2, procsInDirection,periods,0,comm2d);
    MPI_Cart_sub(*comm2d, remain, dimComms);

    remain[0] = 0;
    remain[1] = 1;
    MPI_Cart_sub(*comm2d,remain,dimComms + 1);

    MPI_Datatype tempCol;
    MPI_Type_vector(M, 1, K, MPI_DOUBLE, &tempCol);
    MPI_Type_create_resized(tempCol, 0, sizeof(double), &colType);
    MPI_Type_commit(&colType);

    MPI_Datatype tempRow;
    MPI_Type_vector(1, M, 0, MPI_DOUBLE, &tempRow);
    MPI_Type_create_resized(tempRow, 0, M*sizeof(double), &rowType);
    MPI_Type_commit(&rowType);
}

void setDispls(const int* procsInDirection, MPI_Comm comm2d ){
    inRowSizes.reserve(procSize);
    inColSizes.reserve(procSize);
    inRowDispls.reserve(procSize);
    inColDispls.reserve(procSize);


    inRowDispls.push_back(0);
    inColDispls.push_back(0);
    for( int i = 0; i < procsInDirection[0]; ++i){
        if( i < K % procsInDirection[0]){
            inRowSizes.push_back( K / procsInDirection[0] + 1);
        }
        else {
            inRowSizes.push_back( K / procsInDirection[0]);
        }
        if( i != 0){
            inRowDispls.push_back(inRowDispls[i-1] + inRowSizes[i-1]);
        }
    }
    for( int i = 0; i < procsInDirection[1]; ++i){
        if( i < N % procsInDirection[1]){
            inColSizes.push_back( N / procsInDirection[1] + 1);
        }
        else {
            inColSizes.push_back( N / procsInDirection[1]);
        }
        if( i != 0){
            inColDispls.push_back(inColDispls[i-1] + inColSizes[i-1]);
        }
    }
}

void distributeMatrixes(double matrixA[], double matrixB[], MPI_Comm subComms[], double** localA, double** localB ){
    MPI_Comm_rank(subComms[0], &rankX);
    MPI_Comm_rank(subComms[1], &rankY);
    *localA = new double[M*inColSizes[rankY]];
    *localB = new double[K*inRowSizes[rankX]];
    if(rankX==0){
        MPI_Scatterv(matrixA,inColSizes.data(), inColDispls.data(), rowType, *localA, M*inColSizes[rankY],MPI_DOUBLE,0,subComms[1] );
    }
    if(rankY==0){
        MPI_Scatterv(matrixB,inRowSizes.data(), inRowDispls.data(), colType, *localB, M*inRowSizes[rankX],MPI_DOUBLE,0,subComms[0] );
    }
    MPI_Bcast(*localB, M*inRowSizes[rankX], MPI_DOUBLE,0,subComms[1]);
    MPI_Bcast(*localA, M*inColSizes[rankY], MPI_DOUBLE,0,subComms[0]);
}

void collectResult(double localC[], MPI_Comm subComms[], double resultC[], const int procsInDirection[]){
    double* partC= nullptr;
    localC = transpose(localC, inColSizes[rankY], inRowSizes[rankX]);
    if(rankX==0){
        partC = new double[K*inColSizes[rankY]];
    }
    MPI_Datatype subcol, longSubCol, recRow;
    MPI_Type_vector(inColSizes[rankY],1,inRowSizes[rankX], MPI_DOUBLE, &subcol);
    MPI_Type_create_resized(subcol,0, sizeof(double),&subcol);
    MPI_Type_vector(inColSizes[rankY],1,K,MPI_DOUBLE,&longSubCol);
    MPI_Type_create_resized(subcol,0, sizeof(double),&longSubCol);
    MPI_Type_vector(1,K,K,MPI_DOUBLE,&recRow);
    MPI_Type_commit(&recRow);
    MPI_Type_commit(&subcol);
    MPI_Type_commit(&longSubCol);

MPI_Gatherv(localC, inRowSizes[rankX], subcol, partC, inRowSizes.data(), inRowDispls.data(), longSubCol,0,subComms[0]);
MPI_Type_free(&subcol);
MPI_Type_free(&longSubCol);
MPI_Type_free(&recRow);


    if(rankX==0) {
        MPI_Gatherv(partC, inColSizes[rankY], recRow, resultC, inColSizes.data(), inColDispls.data(), recRow,0,subComms[1]);
        //MPI_Gatherv(partC, recvcounts[rankY], MPI_DOUBLE, resultC, recvcounts,displs, MPI_DOUBLE,0,subComms[1]);
        delete[] partC;
    }
}

double *transpose(double *pDouble, int inColsSize, int inRowsSize) {
    double  buffer[inColsSize*inRowsSize];
    std::copy(pDouble, pDouble+inColsSize*inRowsSize, buffer);
    for(int i = 0; i < inColsSize*inRowsSize; ++i){
        int n = i/inRowsSize;
        int m = i%inRowsSize;
        pDouble[m*inColsSize+n] = buffer[i];
    }
    return pDouble;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm2d;
    MPI_Comm subComms[2];
    int procsInDirection[2] = {0, 0};
    double *matrixA,
            *matrixB,
            *matrixC,
            *localA,
            *localB,
            *localC;
    makeCommEnvironment(&comm2d,subComms,procsInDirection);
    setDispls(procsInDirection, comm2d);
    localC = new double[inColSizes[rankY]*inRowSizes[rankX]];
    std::fill_n(localC,inRowSizes[rankX]*inRowSizes[rankY],0);
    if(0==procRank) {
        initMatrixes(&matrixA, &matrixB, &matrixC);
    }
    distributeMatrixes(matrixA,matrixB, subComms, &localA, &localB);

    for( int i = 0; i < inRowSizes[rankX]; ++i ){
        for( int j = 0 ; j < inColSizes[rankY]; ++j ){
            for( int k = 0; k < M; ++k ){
                localC[ j* inRowSizes[rankX] + i] += localA[j*M + k]*localB[i*M + k];
            }
        }
    }
    collectResult(localC, subComms, matrixC, procsInDirection);
    //transpose(matrixC,N, K);
    delete[] localA;
    delete[] localB;
    delete[] localC;
    if(procRank==0){
        for (int l = 0; l < K; ++l) {
            for( int j = 0; j < N; ++j ){
                std::cout<<matrixC[l*N+j]<<" ";
            }
            std::cout<<std::endl;
        }
        delete [] matrixA;
        delete [] matrixB;
        delete [] matrixC;
    }
    MPI_Finalize();
    return 0;
    /*double matrix[]={1, 2, 3,
                     4,5,6};
    transpose(matrix,2,3);
    for(double a: matrix){
        std::cout<<" "<<a;
    }
    std::cout<<std::endl;*/
}