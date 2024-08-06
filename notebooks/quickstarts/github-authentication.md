# How to push to GitHub from the VEDA JupyterHub

This is a quick how-to guide for pushing to GitHub from the VEDA JupyterHub.

## Configure the veda-hub-github-scoped-creds app

First, you need to install the GitHub auth application in the organization and repository that you will be pushing to. This only needs to be done once per repository.

1. Navigate to [https://github.com/apps/veda-hub-github-scoped-creds](https://github.com/apps/veda-hub-github-scoped-creds).
2. Select `Configure`.
3. Select the organization that contains the repository that you would like to push to.
4. Select either `All repositories` or `Only select repositories` depending on whether you only want to allow pushing to specific repositories. In order for a Hub user to write to those repositories, the repository needs to be added to the application **and** the user needs to have read/write permissions for the repository. The following restrictions define whether you can install the application yourself or if you need to request permission:
    - Organization owners can install GitHub Apps for all repositories.
    - Repository admins can install the application if they only grant access to repositories that they administer.
    - If a requested does not have sufficient permissions, GitHub will send a notification to the organization owner requesting that they install the app.
*. Click save.

## Authenticate using the veda-hub-github-scoped-creds app

Second, you need to authorize the app on the Hub. This needs to be done once every 8 hours.

1. Start an Hub instance.
2. Create a new Jupyter notebook named `gh-scoped-creds.ipynb` in your home directory.
3. Import `import gh_scoped_creds`
4. Run the Jupyter magic command `%ghscopedcreds`, as shown in the following cell.
5. Copy the code that was displayed in the cell output.
6. Navigate to [https://github.com/login/device/select_account](https://github.com/login/device/select_account).
7. Select the account that you want to authorize push access for.
8. Enter the code from the cell output in the Jupyter Notebook.
9. Select `Authorize veda-hub-github-scoped-creds`.

You can re-use the Jupyter notebook created in steps 1-3 each time you need to re-authorize push access.

## Push to repositories

Now, you're all set! You should be able to push to the repositories that you configured in the first portion, as long as you're using the `https` rather than `ssh` protocol. You can set this up by selecting `https` when cloning the repository, or using `git remote set-url` to change from `ssh` to `https`.
