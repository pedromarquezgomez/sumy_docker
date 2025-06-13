from sentence_transformers import SentenceTransformer
 
print("Descargando el modelo 'all-MiniLM-L6-v2'...")
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')
print("Modelo descargado exitosamente en ./model_cache") 