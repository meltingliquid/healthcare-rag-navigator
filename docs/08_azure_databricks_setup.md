# Azure Databricks Setup Guide — Healthcare RAG Capstone Project

Since the entire team is sharing a single Azure account, we are keeping the Databricks setup focused entirely on **execution**. All active coding will happen locally on your machines.

## 1. Create the Databricks Workspace (Team Lead)

The team lead will do this once.

1. Log in to the [Azure Portal](https://portal.azure.com/).
2. In the search bar, type **Azure Databricks** and select it.
3. Click **+ Create** to make a new workspace.
4. Fill in the basics:
   - **Subscription:** Select your active subscription.
   - **Resource Group:** Create a new one named `rg-healthcare-rag`.
   - **Workspace Name:** `dbw-healthcare-capstone`
   - **Region:** Default (e.g., `East US`).
   - **Pricing Tier:** Select **Standard** or **Trial** (if you have standard free credits, Trial is fine for 14 days, otherwise Standard is perfectly adequate).
5. Click **Review + Create**, then **Create**.
6. Wait for the deployment to finish (takes a few minutes), then click **Go to resource** and hit **Launch Workspace**.

---

## 2. Create a Compute Cluster (Team Lead)

To run notebooks, you need compute power. We will build a single cluster that everyone can share to keep costs low.

1. Inside the Databricks Workspace, go to **Compute** on the left menu.
2. Click **Create Compute**.
3. Use these cost-effective settings:
   - **Compute Name:** `Shared-Capstone-Cluster`
   - **Policy:** Unrestricted
   - **Single node** *(Check this box! A multi-node cluster is overkill for building the MVP and burns credits fast).*
   - **Databricks Runtime Version:** Select the latest standard version (e.g., `14.3 LTS (Scala 2.12, Spark 3.5.0)`). Avoid ML versions unless you need specific pre-installed ML libraries immediately.
   - **Node type:** Choose a cheap instance, e.g., `Standard_DS3_v2` (14 GB Memory, 4 Cores).
   - **Terminate after:** Set to **30 minutes** of inactivity. (Crucial to prevent accidental charges if someone forgets to turn it off).
4. Click **Create Compute**.

---

## 3. The Execution Workflow (Local Development to Databricks)

You are **not** collaborating live inside Databricks. Your workflow is:

1. **Code Locally:** Write and test your Python scripts/Jupyter Notebooks on your local machine.
2. **Push to GitHub:** Commit your finished module and push it to the central GitHub repository (e.g., your `feature/*` branch and merge to `main`/`develop`).
3. **Run on Databricks:** You will pull the latest code down into Databricks *only* to run it on the shared cluster. Databricks acts solely as the runtime environment.

---

## 4. Connecting Databricks to GitHub (Pull Only)

To easily get your notebooks into Databricks to run them:

1. In the Databricks Workspace, go to **Workspace > Repos**.
2. Click **Add Repo** and paste your GitHub repository URL.
3. When you need to run a module you just finished locally:
   - Open the Repo in Databricks.
   - Click the Git icon on the left panel or top right.
   - Click **Pull** to get your latest local changes from GitHub.
4. Select the notebook you need to run, attach it to the `Shared-Capstone-Cluster`, and hit **Run**.

---

## 5. Connecting Databricks to AWS S3

Your data lives in S3, but your compute is in Azure Databricks. We need Databricks to read/write to the bucket.

> **Important:** Do NOT commit your AWS keys to GitHub! Your local code should read from an ignored `.env` file. On Databricks, you can use the same pattern or configure Spark directly. Since this is an MVP, a quick configuration at the top of your notebook right before execution works.

### Add this block at the top of Databricks notebooks that access S3:

```python
# Un-comment and fill in with the keys before executing on Databricks
# NEVER COMMIT THESE KEYS BACK TO GITHUB
aws_access_key = "<YOUR_PROJECT2_ACCESS_KEY>"
aws_secret_key = "<YOUR_PROJECT2_SECRET_KEY>"

# Configure Spark to access AWS S3
spark.conf.set("fs.s3a.access.key", aws_access_key)
spark.conf.set("fs.s3a.secret.key", aws_secret_key)
spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")

# AWS assumes the `s3a://` prefix when accessing through Spark
s3_bucket = "s3a://healthcare-rag-capstone/"

# Test the connection:
display(dbutils.fs.ls(s3_bucket))
```

---

## Quick Reference Checklist
- [ ] Azure Resource Group `rg-healthcare-rag` created.
- [ ] Databricks Workspace `dbw-healthcare-capstone` created.
- [ ] Single Node `Shared-Capstone-Cluster` compute created (auto-terminate 30 mins).
- [ ] GitHub Repository linked in Databricks Workspace `Repos`.
- [ ] Local code development -> Push to GitHub -> Pull in Databricks -> Execute workflow established.
