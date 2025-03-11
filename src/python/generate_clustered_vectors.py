#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np

import localconstants

"""
Generate document and query vectors for the vector search task from 
the Cohere/wikipedia-22-12-en-embeddings https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings dataset. 

Usage: 

python src/python/infer_token_vectors_cohere.py -d <num_docs> -q <num_queries>
python src/python/infer_token_vectors_cohere.py -n <filename_prefix> -d <num_docs> -q <num_queries>

For help:
python src/python/infer_token_vectors_cohere.py -h
"""

DATASET_PATH = 'Cohere/wikipedia-22-12-en-embeddings'
DIMENSIONS = 768

def generate_random_embeddings():
  parser = argparse.ArgumentParser(prog='Fetch Wikipedia Cohere Embeddings',
                                     description='Generate document and query vectors for the vector search task '
                                                 'from HuggingFace Cohere/wikipedia-22-12-en-embeddings')
  parser.add_argument('-n', '--name', default='random-clustered',
                      help='Dataset name, used as a filename prefix for generated files.')
  parser.add_argument('-d', '--numDocs', default='500_000', help='Number of documents')
  parser.add_argument('-q', '--numQueries', default='10_000', help='Number of queries')
  parser.add_argument('-c', '--numClusters', default='10', help='Number of clusters')
  parser.add_argument('-r', '--clusterRadius', default='0.00001', help='Cluster radius')
  parser.add_argument('-s', '--seed', default='42', help='Random seed to use for reproducibility')
  args = parser.parse_args()
  print('Fetching Cohere embeddings with the following args: %s' % args)

  doc_file = f"{localconstants.BASE_DIR}/data/{args.name}-docs-{DIMENSIONS}d.vec"
  query_file = f"{localconstants.BASE_DIR}/data/{args.name}-queries-{DIMENSIONS}d.vec"
  num_docs = int(args.numDocs)
  num_queries = int(args.numQueries)
  num_clusters = int(args.numClusters)
  cluster_radius = float(args.clusterRadius)
  seed = int(args.seed)

  for name in (doc_file, query_file):
    print(f'checking if file:{name} exists...')
    if os.path.exists(name):
        raise RuntimeError(f'please remove {name} first')

  embedding_dims = DIMENSIONS
  print(f"embeddings dims: {embedding_dims}")

  # do this in windows, else the RAM usage is crazy (OOME even with 256
  # GB RAM since I think this step makes 2X copy of the dataset?)
  doc_upto = 0
  # window_num_docs = 1000000
  # make it work w/64 GB RAM:
  window_num_docs = 250000
  # window_num_docs = 100000

  if seed is not None:
      np.random.seed(seed)  # Set the seed for reproducibility

  # Generate random cluster centers
  cluster_centers = np.random.rand(num_clusters, embedding_dims)


  # Generate Document Embeddings for clusters
  while doc_upto < num_docs:
    next_doc_upto = min(doc_upto + window_num_docs, num_docs)

    # Assign each vector to a random cluster and apply Gaussian noise
    cluster_indices = np.random.randint(0, num_clusters, size=window_num_docs)  # Random cluster assignments
    ds_embs = cluster_centers[cluster_indices] + np.random.randn(window_num_docs, embedding_dims) * cluster_radius
    # Clip values to ensure they stay in the range [0,1]
    ds_embs = np.clip(ds_embs, 0, 1)

    batch_size = next_doc_upto - doc_upto
    print(f'batch size = {batch_size}')
    embs = np.array(ds_embs, dtype=np.float32)
    print(f'embs: {embs.dtype} {embs.size} {embs.itemsize} {embs.shape}')

    print(f"saving docs[{doc_upto}:{next_doc_upto}] of shape: {embs.shape} to file")
    with open(doc_file, "ab") as out_f:
        embs.tofile(out_f)

    doc_upto = next_doc_upto

  # Write Query Embeddings
  embs_queries = np.random.rand(num_queries, embedding_dims)
  embs_queries = np.array(embs_queries, dtype=np.float32)

  print(f"saving queries of shape: {embs_queries.shape} to file")
  with open(query_file, "w") as out_f_queries:
      embs_queries.tofile(out_f_queries)

  ### check saved datasets
  embs_docs = np.fromfile(doc_file, dtype=np.float32)
  embs_docs = embs_docs.reshape(num_docs, embedding_dims)
  print(f"reading docs of shape: {embs_docs.shape}")
  print(f'{embs_docs[0]}')

  embs_queries = np.fromfile(query_file, dtype=np.float32)
  embs_queries = embs_queries.reshape(num_queries, embedding_dims)
  print(f"reading queries shape: {embs_queries.shape}")
  print(f'{embs_queries[0]}')

if __name__ == '__main__':
  generate_random_embeddings()
