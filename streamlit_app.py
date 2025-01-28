# Streamlit app
import streamlit as st
from main import qa_chain,generate_prompt_by_loanid
def main():
    st.title("Loan Default Risk Assessment")

    # Get the loan ID as input from the user
    loan_id = st.text_input("Enter the loan ID:")

    if st.button("Calculate Default Risk"):
        # Construct the question with the loan ID
        # question = f"What is the default risk associated with loan ID {loan_id}?"
        question = generate_prompt_by_loanid(loan_id,df)
        # Run the QA chain
        try:
            result = qa_chain.invoke({"query": question})
            st.success(result)
        except Exception as e:
            st.error(f"Error encountered: {e}")

if __name__ == "__main__":
    main()