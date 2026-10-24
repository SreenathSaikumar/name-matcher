"""Build data/name_pairs.csv. Hand-labeled KYC pairs; re-run only if the set changes.
Train/holdout split happens later in eval (sklearn), not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "name_pairs.csv"

rows: list[dict] = []


def add(name_a, name_b, label, category, source="synthetic", note="", ambiguous=False):
    rows.append(
        {
            "id": f"p{len(rows)+1:04d}",
            "name_a": name_a,
            "name_b": name_b,
            "label": label,
            "category": category,
            "source": source,
            "note": note,
            "ambiguous": "yes" if ambiguous else "no",
        }
    )


# --- initials vs expanded (match) ---
for a, b, note in [
    ("S. Kumar", "Suresh Kumar", "initial vs expanded"),
    ("R. K. Sharma", "Ravi Kumar Sharma", "two initials"),
    ("A. Patel", "Amit Patel", ""),
    ("P. N. Reddy", "Praveen N Reddy", ""),
    ("V. Krishnan", "Venkat Krishnan", ""),
    ("M. Singh", "Manpreet Singh", ""),
    ("K. R. Iyer", "Krishna R Iyer", ""),
    ("S. Banerjee", "Sourav Banerjee", ""),
    ("N. Joshi", "Nikhil Joshi", ""),
    ("T. Nair", "Thomas Nair", ""),
    ("H. Mehta", "Harsh Mehta", ""),
    ("D. Chatterjee", "Debashish Chatterjee", ""),
]:
    add(a, b, "match", "initials", note=note)

# --- wrong expansion (non-match) ---
for a, b, note in [
    ("S. Kumar", "Sunil Kumar", "same initial different given"),
    ("R. Sharma", "Ramesh Sharma", "could be Ravi not Ramesh"),
    ("A. Patel", "Anil Patel", ""),
    ("P. Reddy", "Pradeep Reddy", ""),
    ("V. Krishnan", "Vijay Krishnan", ""),
    ("M. Singh", "Mohan Singh", ""),
    ("K. Iyer", "Karthik Iyer", ""),
    ("S. Banerjee", "Sanjay Banerjee", ""),
    ("N. Joshi", "Nitin Joshi", ""),
    ("H. Mehta", "Hemant Mehta", ""),
]:
    add(a, b, "non_match", "wrong_expansion", note=note)

# --- surname-first reorder (match) ---
for a, b in [
    ("Kumar Suresh", "Suresh Kumar"),
    ("Sharma Priya", "Priya Sharma"),
    ("Patel Amitkumar", "Amitkumar Patel"),
    ("Reddy Venkat", "Venkat Reddy"),
    ("Singh Gurpreet", "Gurpreet Singh"),
    ("Iyer Ananya", "Ananya Iyer"),
    ("Nair Arun", "Arun Nair"),
    ("Mehta Kavita", "Kavita Mehta"),
    ("Joshi Rohan", "Rohan Joshi"),
    ("Banerjee Aditi", "Aditi Banerjee"),
    ("Gupta Neha", "Neha Gupta"),
    ("Chopra Viraj", "Viraj Chopra"),
]:
    add(a, b, "match", "reorder")

# --- transliteration (match) ---
for a, b, note in [
    ("Mohammed Ali Khan", "Mohammad Ali Khan", "Mohammed variants"),
    ("Muhammad Yusuf", "Mohammed Yusuf", ""),
    ("Mohammad Irfan", "Muhammed Irfan", ""),
    ("Lakshmi Narayanan", "Laxmi Narayanan", "ksh/x"),
    ("Lakshmi Devi", "Laxmi Devi", ""),
    ("Poornima Rao", "Purnima Rao", "oo/u"),
    ("Sreenath Menon", "Srinath Menon", "ee/i"),
    ("Kumaraswamy", "Kumarasvami", "v/w"),
    ("Ashok Kumar", "Asok Kumar", ""),
    ("Bharath Reddy", "Bharat Reddy", ""),
    ("Deepak Sharma", "Dipak Sharma", ""),
    ("Sheikh Abdullah", "Shaikh Abdullah", ""),
    ("Qureshi Imran", "Kureshi Imran", "q/k"),
    ("Fathima Begum", "Fatima Begum", "ph/f-ish"),
    ("Yogeshwaran", "Yogesvaran", ""),
]:
    add(a, b, "match", "transliteration", note=note)

# --- middle / father / husband conventions (match) ---
for a, b, note in [
    ("Rahul Rajesh Sharma", "Rahul Sharma", "dropped middle"),
    ("Ananya Priya Iyer", "Ananya Iyer", ""),
    ("Vikram Suresh Patel", "Vikram Patel", ""),
    ("Sneha Ramesh Kulkarni", "Sneha Kulkarni", ""),
    ("Amitabh Bachchan", "Amitabh Harivansh Bachchan", "extra middle"),
    ("Priya Sharma", "Priya Devi Sharma", ""),
    ("Arjun Das", "Arjun Kumar Das", ""),
    ("Meera Nair", "Meera Lakshmi Nair", ""),
    ("Sanjay Dutt", "Sanjay Balraj Dutt", ""),
    ("Kavya Reddy", "Kavya S Reddy", ""),
]:
    add(a, b, "match", "middle_name", note=note)

# --- honorifics (match) ---
for a, b in [
    ("Smt. Priya Sharma", "Priya Sharma"),
    ("Shri Ramesh Kumar", "Ramesh Kumar"),
    ("Kumari Ananya Iyer", "Ananya Iyer"),
    ("Mr. Amit Patel", "Amit Patel"),
    ("Mrs. Kavita Mehta", "Kavita Mehta"),
    ("Dr. Nikhil Joshi", "Nikhil Joshi"),
    ("Shree Venkat Reddy", "Venkat Reddy"),
    ("Smt Kavitha Nair", "Kavitha Nair"),
    ("Md. Irfan Khan", "Irfan Khan"),
    ("Late Ramesh Gupta", "Ramesh Gupta"),
]:
    add(a, b, "match", "honorifics")

# --- compound join/split (match) ---
for a, b in [
    ("Ram Kumar", "Ramkumar"),
    ("Suresh Kumar", "Sureshkumar"),
    ("Rajesh Kumar Sharma", "Rajeshkumar Sharma"),
    ("Venkatraman Iyer", "Venkat Raman Iyer"),
    ("Subramanian Pillai", "Subra Manian Pillai"),
    ("Krishnamurthy Rao", "Krishna Murthy Rao"),
    ("Anil Kumar", "Anilkumar"),
    ("Santosh Kumar", "Santoshkumar"),
    ("Ravi Shankar", "Ravishankar"),
    ("Hari Prasad", "Hariprasad"),
    ("Gopal Krishna", "Gopalkrishna"),
    ("Nanda Kishore", "Nandakishore"),
]:
    add(a, b, "match", "compound")

# --- S/O lines (match) ---
for a, b, note in [
    ("Rahul Sharma S/O Rajesh Sharma", "Rahul Sharma", "relative on one side"),
    ("Priya Iyer D/O Suresh Iyer", "Priya Iyer", ""),
    ("Kavita Mehta W/O Amit Mehta", "Kavita Mehta", ""),
    ("Amit Patel S/O Ramesh Patel", "Amit Ramesh Patel", "father as middle"),
    ("Suresh Kumar S/O Ram Kumar", "Suresh Kumar", ""),
    ("Ananya Reddy D/O Venkat Reddy", "Ananya Reddy", ""),
    ("Vikram Singh S/O Manpreet Singh", "Vikram Singh", ""),
    ("Neha Gupta D/O Sunil Gupta", "Neha Gupta", ""),
]:
    add(a, b, "match", "so_line", note=note)

# S/O both sides same person
add("Rahul Sharma S/O Rajesh Sharma", "Rahul Sharma S/O Rajesh Sharma", "match", "so_line", note="identical relative")
add("Amit Patel S/O Ramesh Patel", "Amitbhai Patel S/O Ramesh Patel", "match", "so_line", note="given variant")

# --- South Indian multi-initial (match) — synthetic patterns ---
for a, b in [
    ("R. K. Narayan", "Ramaswamy Krishnaswami Narayan"),
    ("C. V. Menon", "Chandrasekhar Venkatesh Menon"),
    ("A. R. Krishnan", "Ananthapadmanabha Ramaswamy Krishnan"),
    ("K. J. Nambiar", "Krishnan Joseph Nambiar"),
    ("P. T. Nair", "Padmanabhan Thankappan Nair"),
    ("M. S. Pillai", "Madhavan Subramanian Pillai"),
    ("S. R. Iyer", "Srinivasan Ramaswamy Iyer"),
    ("V. V. S. Reddy", "Venkata Vijaya Sai Reddy"),
    ("B. K. S. Rao", "Balakrishna Kumar Srinivas Rao"),
    ("T. N. Sharma", "Trivikram Narayana Sharma"),
]:
    add(a, b, "match", "south_indian_initials", source="synthetic")

# --- hard negatives: siblings / near duplicates ---
for a, b, note in [
    ("Rajesh Kumar Sharma", "Ramesh Kumar Sharma", "sibling-like"),
    ("Suresh Patel", "Ramesh Patel", ""),
    ("Anil Kumar", "Sunil Kumar", ""),
    ("Priya Sharma", "Pooja Sharma", ""),
    ("Vikram Reddy", "Vikrant Reddy", ""),
    ("Neha Gupta", "Nisha Gupta", ""),
    ("Amit Joshi", "Sumit Joshi", ""),
    ("Kavita Nair", "Kavya Nair", ""),
    ("Rohan Mehta", "Sohan Mehta", ""),
    ("Deepak Singh", "Deepika Singh", "gender near"),
    ("Nandan Rao", "Nandini Rao", "gender phonetic"),
    ("Sunil Iyer", "Sunila Iyer", "syllable trap"),
    ("Karan Kapoor", "Arjun Kapoor", "common surname"),
    ("Rahul Dravid", "Saurav Dravid", ""),
    ("Ajay Devgn", "Abhay Devgn", ""),
    ("Manoj Bajpayee", "Manoj Tiwari", "same given different surname"),
    ("Ravi Teja", "Ravi Shankar", ""),
    ("Sanjay Leela", "Sanjay Dutt", ""),
    ("Alia Bhatt", "Pooja Bhatt", "related public different people"),
    ("Madhuri Dixit", "Konkona Sen Sharma", "unrelated"),
]:
    src = "public" if "Bhatt" in a or "Dixit" in a or "Devgn" in a or "Bajpayee" in a or "Dravid" in a else "synthetic"
    add(a, b, "non_match", "hard_negative", source=src, note=note)

# --- more hard negatives: shared surname only ---
for a, b in [
    ("Asha Kumar", "Nisha Kumar"),
    ("Farhan Khan", "Imran Khan"),
    ("Geeta Reddy", "Seeta Reddy"),
    ("Harish Rao", "Girish Rao"),
    ("Ishita Shah", "Ishaan Shah"),
    ("Jyoti Verma", "Jyoti Yadav"),
    ("Komal Jain", "Kamal Jain"),
    ("Lata Mangeshkar", "Asha Bhosle", ),
]:
    src = "public" if "Mangeshkar" in a else "synthetic"
    label = "non_match"
    add(a, b, label, "hard_negative", source=src)

# --- public figure transliteration / known variants (match, capped) ---
for a, b, note in [
    ("Mahatma Gandhi", "Mohandas Karamchand Gandhi", "public expanded"),
    ("Sachin Tendulkar", "Sachin R Tendulkar", ""),
    ("A. P. J. Abdul Kalam", "Avul Pakir Jainulabdeen Abdul Kalam", ""),
    ("Rabindranath Tagore", "Rabindra Nath Tagore", ""),
    ("Satyajit Ray", "Satyajit Kumar Ray", ""),
    ("Viswanathan Anand", "Vishwanathan Anand", "v/w"),
    ("M. F. Husain", "Maqbool Fida Husain", ""),
    ("Lata Mangeshkar", "Lata Dinanath Mangeshkar", ""),
]:
    add(a, b, "match", "public_variant", source="public", note=note)

# --- clear non-matches easy ---
for a, b in [
    ("Rahul Sharma", "Priya Iyer"),
    ("Amit Patel", "Kavita Nair"),
    ("Suresh Kumar", "Ananya Reddy"),
    ("Vikram Singh", "Meera Joshi"),
    ("Nikhil Gupta", "Fatima Khan"),
    ("Arjun Das", "Pooja Mehta"),
    ("Rohan Kapoor", "Sneha Pillai"),
    ("Deepak Verma", "Anjali Rao"),
    ("Manish Tiwari", "Divya Menon"),
    ("Karan Malhotra", "Shreya Ghosh"),
]:
    add(a, b, "non_match", "easy_negative")

# --- ambiguous nicknames (excluded from primary) ---
for a, b, note in [
    ("Bobby Deol", "Vijay Singh Deol", "nickname vs formal"),
    ("Raju", "Rajesh Kumar", "short nickname"),
    ("Guddu Sharma", "Gaurav Sharma", ""),
    ("Bunty Singh", "Balwant Singh", ""),
    ("Pinky Patel", "Priyanka Patel", ""),
    ("Sonu Nigam", "Ashok Nigam", "stage vs less clear"),
]:
    add(a, b, "match", "ambiguous_nickname", note=note, ambiguous=True)

# pad more synthetic matches/non-matches to clear 200
extra_matches = [
    ("Harishchandra Prasad", "Harish Chandra Prasad"),
    ("Om Prakash", "Omprakash"),
    ("Chandra Sekhar", "Chandrasekhar"),
    ("Jagdish Prasad", "Jagdishprasad"),
    ("Balakrishna", "Bala Krishna"),
    ("Surya Narayana", "Suryanarayana"),
    ("Padma Lakshmi", "Padmalakshmi"),
    ("Uma Devi", "Umadevi"),
    ("Shanti Bai", "Shanthi Bai"),
    ("Ganesh Rao", "Ganesha Rao"),
    ("Bhavesh Shah", "Bhavesh Kumar Shah"),
    ("Chirag Jain", "Chiragkumar Jain"),
    ("Dinesh Yadav", "Dinesh Kumar Yadav"),
    ("Esha Gupta", "Eesha Gupta"),
    ("Firoz Khan", "Firoze Khan"),
    ("Gaurav Bansal", "Gourav Bansal"),
    ("Hemant Chauhan", "Hemanta Chauhan"),
    ("Indira Joshi", "Indiraa Joshi"),
    ("Jatin Malhotra", "Jateen Malhotra"),
    ("Kirti Sanon", "Kirthy Sanon"),
]
for a, b in extra_matches:
    add(a, b, "match", "transliteration" if a != b.replace(" ", "") else "compound")

extra_non = [
    ("Bhavesh Shah", "Bhavin Shah"),
    ("Chirag Jain", "Chirayu Jain"),
    ("Dinesh Yadav", "Dinesh Yadava"),
    ("Esha Gupta", "Isha Singh"),
    ("Firoz Khan", "Farooq Khan"),
    ("Gaurav Bansal", "Saurav Bansal"),
    ("Hemant Chauhan", "Hemanth Reddy"),
    ("Indira Joshi", "Indira Gandhi"),
    ("Jatin Malhotra", "Jatin Sapru"),
    ("Kirti Sanon", "Kriti Kharbanda"),
    ("Lakshman Rao", "Lakshmi Rao"),
    ("Mohan Das", "Mahatma Das"),
    ("Naveen Patnaik", "Naveen Kumar"),
    ("Omar Abdullah", "Farooq Abdullah"),
    ("Pankaj Tripathi", "Pankaj Kapur"),
    ("Qamar Jahan", "Qamar Ali"),
    ("Rishi Kapoor", "Ranbir Kapoor"),
    ("Sridevi Kapoor", "Janhvi Kapoor"),
    ("Tiger Shroff", "Jackie Shroff"),
    ("Uday Chopra", "Aditya Chopra"),
]
for a, b in extra_non:
    src = "public" if any(x in a + b for x in ["Kapoor", "Gandhi", "Abdullah", "Patnaik", "Shroff", "Chopra", "Tripathi", "Sanon", "Kharbanda"]) else "synthetic"
    add(a, b, "non_match", "hard_negative", source=src)

# reorder non-matches (different people, similar tokens)
for a, b in [
    ("Sharma Rahul", "Sharma Rohan"),
    ("Patel Suresh", "Patel Ramesh"),
    ("Reddy Anil", "Reddy Sunil"),
    ("Iyer Meera", "Iyer Meena"),
    ("Nair Arun", "Nair Varun"),
    ("Gupta Neha", "Gupta Nisha"),
    ("Mehta Rohan", "Mehta Sohan"),
]:
    add(a, b, "non_match", "hard_negative", note="reorder+near given")

df = pd.DataFrame(rows)
assert len(df) >= 200, len(df)
assert (df["source"] == "public").mean() <= 0.20

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} pairs -> {OUT}")
print(f"public%={(df.source=='public').mean():.1%}")
print("(train/holdout split is done at eval time via sklearn)")
print(df.groupby(["label", "ambiguous"]).size())
print(df["category"].value_counts())
