#ifndef _exchange_externals_hpp_
#define _exchange_externals_hpp_

//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#include <cstdlib>
#include <iostream>

#define TAUSCH_CUDA
#include "../utils/tausch.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <outstream.hpp>

#include <TypeTraits.hpp>

namespace miniFE {

template<typename Scalar, typename Index> 
  __global__ void copyElementsToBuffer(Scalar *src, Scalar *dst, Index *indices, int N) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    int idx=indices[i];
    dst[i]=__ldg(src+idx);
  }
}

template<typename MatrixType,
         typename VectorType>
void setup_tausch(MatrixType& A, VectorType& x, Tausch *tausch) {

  typedef typename MatrixType::ScalarType Scalar;

  // compose recv information
  // these are all stored in consecutive memory at the end of the data array

  int offset = A.rows.size();

  for(int i=0; i < A.neighbors.size(); ++i) {

    int n_recv = A.recv_length[i];

    std::vector<std::array<int,4> > indices;
    indices.push_back({offset, n_recv, 1, 1});

    tausch->addRecvHaloInfo(indices, sizeof(Scalar), 1);
#ifdef GPUDIRECT
    tausch->setRecvCommunicationStrategy(i, Tausch::Communication::CUDAAwareMPI);
#endif

    offset += n_recv;

  }

  // compose send information
  // these are stored at the locations specified by elements_to_send

  offset = 0;

  for(int i=0; i < A.neighbors.size(); ++i) {
    int n_send = A.send_length[i];

    std::vector<int> indices;
    for(int j = offset; j < offset+n_send; ++j)
      indices.push_back(A.elements_to_send[j]);

    tausch->addSendHaloInfo(indices, sizeof(Scalar), 1);

    offset += n_send;

  }

}

template<typename MatrixType,
         typename VectorType>
void
exchange_externals(MatrixType& A,
                   VectorType& x,
                   Tausch *tausch)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "entering exchange_externals\n";
#endif

  int numprocs = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  if (numprocs < 2) return;
  
  typedef typename MatrixType::ScalarType Scalar;

  int MPI_MY_TAG = 99;

  std::vector<MPI_Request>& request = A.request;

  //
  // Post receives first
  //

  for(int i=0; i<A.neighbors.size(); ++i) {
    request[i] = tausch->recv(i, MPI_MY_TAG, A.neighbors[i], 0, false);
  }

  //
  // Pack and send to each neighbor
  //

  for(int i=0; i<A.neighbors.size(); ++i) {
    tausch->packSendBufferCUDA(i, 0, static_cast<Scalar*>(thrust::raw_pointer_cast(&(x.d_coefs[0]))));
    tausch->send(i, MPI_MY_TAG, A.neighbors[i]);
  }

  //
  // Recv and unpack from each neighbor
  //

  MPI_Status status;
  for(int i=0; i<A.neighbors.size(); ++i) {
    if(MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    tausch->unpackRecvBuffer(i, 0, &(x.coefs[0]));
  }

#ifdef MINIFE_DEBUG
  os << "leaving exchange_externals"<<std::endl;
#endif

//endif HAVE_MPI
#endif
}

#ifdef HAVE_MPI
static std::vector<MPI_Request> exch_ext_requests;
#endif

template<typename MatrixType,
         typename VectorType>
void
begin_exchange_externals(MatrixType& A,
                         VectorType& x,
                         Tausch *tausch)
{
#ifdef HAVE_MPI

  int numprocs = 1, myproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  if (numprocs < 2) return;

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //

  int MPI_MY_TAG = 99;

  exch_ext_requests.resize(A.neighbors.size()*2);

  // Post receives

  for(int i=0; i<A.neighbors.size(); ++i) {
    exch_ext_requests[i] = tausch->recv(i, MPI_MY_TAG, A.neighbors[i], 0, false);
  }

  //
  // Pack and send to each neighbor
  //

  for(int i=0; i<A.neighbors.size(); ++i) {
    tausch->packSendBufferCUDA(i, 0, static_cast<Scalar*>(thrust::raw_pointer_cast(&(x.d_coefs[0]))));
    tausch->send(i, MPI_MY_TAG, A.neighbors[i]);
  }

#endif

}

template<typename MatrixType,
         typename VectorType>
inline
void
finish_exchange_externals(MatrixType &A, VectorType &x, Tausch *tausch)
{
#ifdef HAVE_MP

  //
  // Recv and unpack from each neighbor
  //

  MPI_Status status;
  for(int i=0; i<A.neighbors.size(); ++i) {
    if(MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    tausch->unpackRecvBuffer(i, 0, &(x.coefs[0]));
  }

//endif HAVE_MPI
#endif
}

}//namespace miniFE

#endif

