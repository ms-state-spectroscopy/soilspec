import pandas as pd

if __name__ == "__main__":
    df1 = pd.DataFrame({"Name": ["Pete", "John", "Max"], "Age": [19, 30, 24]})
    df2 = pd.DataFrame(
        {
            "Name": ["Pete", "Pete", "John", "Max", "Max"],
            "Subject": ["Math", "History", "English", "History", "Math"],
            "Grade": [90, 100, 90, 90, 80],
        }
    )
    df3 = pd.merge(df1, df2, how="right", on="Name")
    print(df1)
    print(df2)
    print(df3)

    df1 = pd.read_csv("asd_csv.csv").set_index("sample_id")
    df2 = pd.read_csv("merged_xlsx.csv").set_index("sample_id")
    df3 = (
        pd.merge(
            df1,
            df2,
            how="right",
            left_index=True,
            right_index=True,
            suffixes=("", "_y"),
        )
        .reset_index()
        .drop_duplicates(subset=["sample_id", "trial"])
        .set_index("sample_id")
    )

    df3.drop(df3.filter(regex="_y$").columns, axis=1, inplace=True)
    df3.drop(df3.filter(regex="_x$").columns, axis=1, inplace=True)

    print(df1)
    print(df2)
    print(df3)
    df3.to_csv("merge_test_result.csv")
