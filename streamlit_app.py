import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'movies_2024.csv'
df = pd.read_csv(file_path)

# Filter necessary columns
budget_revenue_df = df[['budget', 'revenue']].dropna()

# Streamlit App
st.title("Budget vs Revenue Analysis")

# Create a scatter plot to visualize the relationship between budget and revenue
st.write("### Scatter Plot of Budget vs Revenue")
fig, ax = plt.subplots()
ax.scatter(budget_revenue_df['budget'], budget_revenue_df['revenue'], alpha=0.5)
ax.set_xlabel('Budget (in dollars)')
ax.set_ylabel('Revenue (in dollars)')
ax.set_title('Budget vs Revenue')

st.pyplot(fig)

# Display Dataframe
st.write("### Data used for visualization")
st.write(budget_revenue_df)
