
def create_prompt(data_point):
    prompt =  f"""Predict the default probability for this user given LoanID: {data_point['LoanID']}.
Details:
    - Age: {data_point['Age']}
    - Income: {data_point['Income']}
    - Loan Amount: {data_point['LoanAmount']}
    - Credit Score: {data_point['CreditScore']}
    - Months Employed: {data_point['MonthsEmployed']}
    - Number of Credit Lines: {data_point['NumCreditLines']}
    - Interest Rate: {data_point['InterestRate']}
    - Loan Term: {data_point['LoanTerm']}
    - Debt-to-Income Ratio: {data_point['DTIRatio']}
    - Education: {data_point['Education']}
    - Employment Type: {data_point['EmploymentType']}
    - Marital Status: {data_point['MaritalStatus']}
    - Has Mortgage: {data_point['HasMortgage']}
    - Has Dependents: {data_point['HasDependents']}
    - Loan Purpose: {data_point['LoanPurpose']}
    - Has Co-Signer: {data_point['HasCoSigner']}
    calculate it when the LoanId is given as the input
    """

    return prompt
def generate_prompt_by_loanid(loan_id, df):
    # Filter the dataframe for the provided LoanID
    data_point = df[df['LoanID'] == loan_id]    
    # Check if LoanID exists
    if data_point.empty:
        return f"No record found for LoanID: {loan_id}"   
    # Generate prompt for the filtered data
    prompt = create_prompt(data_point.iloc[0])  # Convert row to series
    return prompt

if __name__ == "__main__":
    generate_prompt_by_loanid()