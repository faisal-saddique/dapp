import streamlit as st
from pinecone_utils.all_pinecone_utils import (del_all_vecs_in_a_namespace,del_old_ind_n_create_new_one,describe_index_stats,list_all_indexes)

st.subheader("Pinecone Index stats")
describe_index_stats()
st.divider()

st.subheader("Pinecone Indices")
list_all_indexes()
st.divider()

st.subheader("Destructive Operations")
if st.button("Recreate Index"):
    del_old_ind_n_create_new_one()