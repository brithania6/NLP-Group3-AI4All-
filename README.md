import numpy as np

mock_data = {
    "Job Title": ["Software Engineer", "Data Scientist", "Product Manager", None, "UX Designer"],
    "Company": ["Google", "Amazon", "Meta", "Apple", None],
    "Location": ["New York, NY", "Seattle, WA", None, "San Francisco, CA", "Austin, TX"],
    "Posted Date": ["2024-10-15", "2024-11-01", "InvalidDate", "2024-10-25", "2024-11-10"],
    "Skills": ["Python, Machine Learning", None, "Agile, Product Strategy", "Swift, Objective-C", "Figma, UX Research"],
    "Salary": ["120000-140000", None, "130000-150000", "115000-135000", "105000-125000"]
}

mock_df = pd.DataFrame(mock_data)

mock_df.info(), mock_df.head()
