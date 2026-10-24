import math

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    def eqdist(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    centroids = list(initial_centroids)
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [eqdist(point, centroid) for centroid in centroids]
            closest_centroid_idx = distances.index(min(distances))
            clusters[closest_centroid_idx].append(point)
            
        
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                mean_point = tuple(
                    sum(coord) / len(clusters[i]) for coord in zip(*clusters[i])
                )
                new_centroids.append(mean_point)
            else:
                new_centroids.append(centroids[i])
                
        if centroids == new_centroids:
            break
            
        centroids = new_centroids

    final_centroids = [
        tuple(round(coord, 4) for coord in centroid) for centroid in centroids
    ]
    
    return final_centroids