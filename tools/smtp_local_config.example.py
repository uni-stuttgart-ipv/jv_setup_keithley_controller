"""Copy me to `smtp_local_config.py` and fill in the real values.

    copy tools\\smtp_local_config.example.py tools\\smtp_local_config.py

`smtp_local_config.py` is in .gitignore, so git will not pick it up even
with `git add .`. THIS file -- the example -- is tracked, so never put a
real password in it.

With that file in place:

    python tools\\setup_smtp_credential.py --test

runs with no arguments and no prompts. It reads these values, stores the
password in Windows Credential Manager, and writes the non-secret settings
where the application looks for them.

Keep this file off the lab PCs once they are commissioned, or keep it only
where the admin account can read it.
"""

SMTP_HOST = "smtp.uni-stuttgart.de"
SMTP_PORT = 587
SMTP_USER = ""                  # the no-reply account name
SMTP_FROM_ADDR = ""             # address the OTP mail comes from
SMTP_FROM_NAME = "SolarJV Analyzer"

# The only secret. Fine here, because this file never reaches git.
SMTP_PASSWORD = ""
