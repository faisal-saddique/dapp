import pinecone
from dotenv import load_dotenv
import os
import time 
import streamlit as st

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

index = pinecone.Index(os.getenv("PINECONE_INDEX"))

def del_all_vecs_in_a_namespace():
    namespace = ""
    st.write(f"Deleting all vectors in namespace: {namespace}")
    index.delete(delete_all=True, namespace=namespace)
    st.write("All vectors deleted!")


def del_old_ind_n_create_new_one():
    pinecone.delete_index(os.getenv("PINECONE_INDEX"))
    st.info("Old index deleted.")
    time.sleep(6)
    # pinecone.create_index(os.getenv("PINECONE_INDEX"), dimension=1536, 
    #                     metric='cosine', 
    #                     pods=1, 
    #                     replicas=1, 
    #                     pod_type='p1.x1')

    # create the index

    pinecone.create_index(
        name=os.getenv("PINECONE_INDEX"),
        dimension=1536,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        pod_type="'p1.x1'",
        metadata_config={"indexed": []},  # see explaination above
    )
    
    st.success("New index created.")

def describe_index_stats():
    stats = index.describe_index_stats()
    namespaces = stats['namespaces']
    for key, value in zip(namespaces.keys(), namespaces.values()):
        if key != "":
            st.write(f"Namespace: {key} => Vectors: {value}")
        else:
            st.write(f"Namespace: No namespace => Vectors: {value}")
    st.write(f"Total vectors: {stats['total_vector_count']}")

def list_all_indexes():
    st.write(pinecone.list_indexes())