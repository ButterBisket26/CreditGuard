CREATE TABLE loans (
    loan_id            SERIAL PRIMARY KEY,
    loan_amnt          NUMERIC(10,2) NOT NULL,
    term               SMALLINT NOT NULL,
    int_rate           NUMERIC(5,2) NOT NULL,
    grade              CHAR(1) NOT NULL,
    sub_grade          VARCHAR(3) NOT NULL,
    emp_length_years   SMALLINT,
    emp_length_missing BOOLEAN NOT NULL,
    home_ownership     VARCHAR(20),
    annual_inc         NUMERIC(12,2),
    purpose            VARCHAR(30),
    dti                NUMERIC(6,2),
    dti_missing        BOOLEAN NOT NULL,
    delinq_2yrs        SMALLINT,
    open_acc           SMALLINT,
    pub_rec            SMALLINT,
    revol_bal          NUMERIC(12,2),
    revol_util         NUMERIC(6,2),
    revol_util_missing BOOLEAN NOT NULL,
    total_acc          SMALLINT,
    issue_d            DATE NOT NULL,
    issue_year         SMALLINT NOT NULL,
    issue_quarter      SMALLINT NOT NULL,
    addr_state         CHAR(2),
    application_type   VARCHAR(20),
    default_flag       BOOLEAN NOT NULL
);

-- Indexes for the SQL analysis layer we'll build next
CREATE INDEX idx_loans_issue_year ON loans(issue_year);
CREATE INDEX idx_loans_grade ON loans(grade);
CREATE INDEX idx_loans_addr_state ON loans(addr_state);
CREATE INDEX idx_loans_default_flag ON loans(default_flag);