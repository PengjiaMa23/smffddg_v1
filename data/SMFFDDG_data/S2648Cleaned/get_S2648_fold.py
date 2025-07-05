import pandas as pd
import numpy as np
import os
from collections import defaultdict

# File paths
input_csv = "./data/SMFFDDG_data/S2648Cleaned/S2648.csv"
output_csv = "./data/SMFFDDG_data/S2648_with_cluster_fold.csv"
cluster_file = "./data/original_data/S2648Cleaned/S2648.cluster.25"

# Read the original data
df = pd.read_csv(input_csv)

# 1. Process the clustering file
cluster_map = {}  # Protein ID -> Cluster ID
cluster_proteins = defaultdict(list)  # Cluster ID -> List of proteins
cluster_id_counter = 0

with open(cluster_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        cluster_size = int(parts[0])
        proteins = parts[1:]
        
        # Process each protein ID: convert to lowercase (except for the chain identifier)
        processed_proteins = []
        for protein in proteins:
            # Keep the chain identifier (the last character) in uppercase, and convert the rest to lowercase
            if len(protein) > 1:
                base = protein[:-1].lower()
                chain = protein[-1].upper()
                processed_protein = base + chain
                processed_proteins.append(processed_protein)
        
        # Assign a cluster ID and record the mapping for the current cluster
        for protein in processed_proteins:
            cluster_map[protein] = cluster_id_counter
        cluster_proteins[cluster_id_counter] = processed_proteins
        
        cluster_id_counter += 1

# 2. Extract protein IDs from mutation data and convert them
def extract_and_convert_protein_id(mut_id):
    """Extract protein ID from mutation ID and convert it to a standardized format"""
    protein_id = mut_id.split('_')[0]
    if len(protein_id) > 1:
        base = protein_id[:-1].lower()
        chain = protein_id[-1].upper()
        return base + chain
    return protein_id

# Apply the conversion function
df['protein_id'] = df['id'].apply(extract_and_convert_protein_id)

# 3. Map proteins to clusters
df['cluster_id'] = df['protein_id'].map(cluster_map)

# Handle unclustered proteins (assign new cluster IDs)
max_cluster_id = max(cluster_proteins.keys()) if cluster_proteins else -1
unclustered_mask = df['cluster_id'].isna()
num_unclustered = unclustered_mask.sum()

if num_unclustered > 0:
    print(f"Found {num_unclustered} unclustered proteins, assigning new clusters for them...")
    
    # Create a separate cluster for each unclustered protein
    unclustered_proteins = df.loc[unclustered_mask, 'protein_id'].unique()
    for protein in unclustered_proteins:
        max_cluster_id += 1
        cluster_map[protein] = max_cluster_id
        cluster_proteins[max_cluster_id] = [protein]
    
    # Update cluster IDs
    df['cluster_id'] = df['protein_id'].map(cluster_map)

# 4. Stratify clusters into five folds
# Count mutations in each cluster
cluster_mutation_counts = df.groupby('cluster_id').size()

# Sort clusters by size (largest to smallest)
sorted_clusters = cluster_mutation_counts.sort_values(ascending=False).index.tolist()

# Initialize lists of clusters and mutation counts for each fold
folds = [[] for _ in range(5)]
fold_counts = [0] * 5

# Assign clusters to folds (greedy algorithm to balance fold sizes)
for cluster_id in sorted_clusters:
    # Find the fold with the current smallest count
    min_fold = np.argmin(fold_counts)
    
    # Assign the current cluster to this fold
    folds[min_fold].append(cluster_id)
    fold_counts[min_fold] += cluster_mutation_counts[cluster_id]

# 5. Create a mapping from clusters to folds
cluster_to_fold = {}
for fold_idx, cluster_list in enumerate(folds):
    for cluster_id in cluster_list:
        cluster_to_fold[cluster_id] = fold_idx

# 6. Assign folds to each mutation
df['fold'] = df['cluster_id'].map(cluster_to_fold)

# 7. Clean up temporary columns and save the result
result_df = df.drop(columns=['protein_id', 'cluster_id'])
result_df.to_csv(output_csv, index=False)

# Print statistics
print(f"Processing complete! Results saved to: {output_csv}")
print(f"Total number of mutations: {len(df)}")
print(f"Number of clusters used: {len(cluster_proteins)}")
print("Distribution of mutations across folds:")
for fold_idx in range(5):
    num_mutations = (df['fold'] == fold_idx).sum()
    num_clusters = len(folds[fold_idx])
    print(f"Fold {fold_idx}: {num_mutations} mutations, {num_clusters} clusters")

print("\nCluster size distribution:")
print(cluster_mutation_counts.describe())