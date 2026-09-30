"use strict";

/*
 * Data Cleaning Workflow Engine
 *
 * This Node.js program models a cleaning pipeline through an event-driven
 * workflow. It intentionally treats missing values, duplicates, outliers,
 * and inconsistent values as different data-quality problems.
 *
 * Run with:
 *   node data_cleaning.js
 */

const fs = require("fs");
const path = require("path");
const EventEmitter = require("events");

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

function createTransactions() {
    return [
        {
            id: "TX2001",
            customer: "  Priya Sharma ",
            email: "PRIYA.SHARMA@EXAMPLE.COM",
            country: "IND",
            state: "UP",
            category: "electronics",
            amount: "12500",
            date: "2026-09-01"
        },
        {
            id: "TX2002",
            customer: "Rahul Verma",
            email: "rahul.verma@example.com",
            country: "India",
            state: "uttar pradesh",
            category: "Electronics",
            amount: "13,250",
            date: "01/09/2026"
        },
        {
            id: "TX2003",
            customer: "Neha Singh",
            email: "neha.singh@example.com",
            country: "IN",
            state: "U.P.",
            category: "Home and Kitchen",
            amount: "",
            date: "2026-09-03"
        },
        {
            id: "TX2004",
            customer: "Amit Kumar",
            email: "amit.kumar@example.com",
            country: "India",
            state: "Delhi",
            category: "home & kitchen",
            amount: "8900",
            date: null
        },
        {
            id: "TX2005",
            customer: "Sara Khan",
            email: "sara.khan@example.com",
            country: "India",
            state: "delhi",
            category: "HOME & KITCHEN",
            amount: "9100",
            date: "2026-09-05"
        },
        {
            id: "TX2006",
            customer: "Vikram Rao",
            email: "vikram.rao@example.com",
            country: "United States",
            state: "California",
            category: "Electronics",
            amount: "1500000",
            date: "2026-09-06"
        },
        {
            id: "TX2007",
            customer: "Meera Joshi",
            email: "MEERA.JOSHI@example.com",
            country: "USA",
            state: "CA",
            category: "electronics",
            amount: "11750",
            date: "2026-09-07"
        },
        {
            id: "TX2008",
            customer: "Arjun Patel",
            email: "arjun.patel@example.com",
            country: "INDIA",
            state: "Gujarat",
            category: "Electronics",
            amount: "12100",
            date: "07-09-2026"
        },
        {
            id: "TX2009",
            customer: "Kabir Mehta",
            email: "kabir.mehta@example.com",
            country: "IN",
            state: "MH",
            category: "Grocery",
            amount: "2150",
            date: "2026-09-08"
        },
        {
            id: "TX2009",
            customer: "Kabir Mehta",
            email: "kabir.mehta@example.com",
            country: "IN",
            state: "MH",
            category: "Grocery",
            amount: "2150",
            date: "2026-09-08"
        }
    ];
}

// ---------------------------------------------------------------------------
// Cleaning utilities
// ---------------------------------------------------------------------------

function isMissing(value) {
    return value === null ||
        value === undefined ||
        (typeof value === "string" && value.trim() === "");
}

function normalizeText(value) {
    if (isMissing(value)) {
        return null;
    }

    return String(value).trim().replace(/\s+/g, " ");
}

function normalizeEmail(value) {
    const normalized = normalizeText(value);
    return normalized === null ? null : normalized.toLowerCase();
}

function parseAmount(value) {
    if (isMissing(value)) {
        return null;
    }

    const cleaned = String(value)
        .replace(/,/g, "")
        .replace(/[₹$€£]/g, "")
        .trim();

    const number = Number(cleaned);

    return Number.isFinite(number) ? number : null;
}

function canonicalize(value, mapping) {
    const normalized = normalizeText(value);

    if (normalized === null) {
        return null;
    }

    return mapping[normalized.toLowerCase()] ?? normalized;
}

const COUNTRY_MAP = Object.freeze({
    "india": "India",
    "ind": "India",
    "in": "India",
    "united states": "United States",
    "usa": "United States",
    "us": "United States"
});

const STATE_MAP = Object.freeze({
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "uttar pradesh": "Uttar Pradesh",
    "delhi": "Delhi",
    "ca": "California",
    "california": "California",
    "gujarat": "Gujarat",
    "mh": "Maharashtra"
});

const CATEGORY_MAP = Object.freeze({
    "electronics": "Electronics",
    "home and kitchen": "Home & Kitchen",
    "home & kitchen": "Home & Kitchen",
    "grocery": "Grocery",
    "groceries": "Grocery"
});

function normalizeDate(value) {
    if (isMissing(value)) {
        return null;
    }

    const text = String(value).trim();

    if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
        return text;
    }

    const slashMatch = text.match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
    const dashMatch = text.match(/^(\d{2})-(\d{2})-(\d{4})$/);
    const match = slashMatch || dashMatch;

    if (match) {
        const [, day, month, year] = match;
        return `${year}-${month}-${day}`;
    }

    throw new Error(`Unsupported date representation: ${value}`);
}

// ---------------------------------------------------------------------------
// Event-driven cleaning engine
// ---------------------------------------------------------------------------

class CleaningEngine extends EventEmitter {
    constructor(records) {
        super();

        this.records = records.map(record => ({ ...record }));
        this.audit = [];
        this.metrics = {
            missingHandled: 0,
            duplicatesRemoved: 0,
            inconsistentValues: 0,
            outliersFlagged: 0
        };
    }

    recordAudit(recordId, field, issue, before, after, action, reason) {
        const entry = {
            timestamp: new Date().toISOString(),
            recordId,
            field,
            issue,
            before,
            after,
            action,
            reason
        };

        this.audit.push(entry);
        this.emit("issue", entry);
    }

    normalizeValues() {
        for (const record of this.records) {
            const changes = {
                customer: normalizeText(record.customer),
                email: normalizeEmail(record.email),
                country: canonicalize(record.country, COUNTRY_MAP),
                state: canonicalize(record.state, STATE_MAP),
                category: canonicalize(record.category, CATEGORY_MAP)
            };

            for (const [field, value] of Object.entries(changes)) {
                if (record[field] !== value) {
                    this.recordAudit(
                        record.id,
                        field,
                        "inconsistent_value",
                        record[field],
                        value,
                        "canonicalize",
                        "Converted an equivalent representation to a domain-defined canonical value."
                    );

                    record[field] = value;
                    this.metrics.inconsistentValues++;
                }
            }

            if (!isMissing(record.amount)) {
                const numericAmount = parseAmount(record.amount);

                if (numericAmount !== null &&
                    record.amount !== numericAmount) {
                    this.recordAudit(
                        record.id,
                        "amount",
                        "inconsistent_value",
                        record.amount,
                        numericAmount,
                        "normalize_numeric",
                        "Removed presentation-level numeric separators."
                    );

                    record.amount = numericAmount;
                    this.metrics.inconsistentValues++;
                }
            }

            if (!isMissing(record.date)) {
                try {
                    const normalizedDate = normalizeDate(record.date);

                    if (normalizedDate !== record.date) {
                        this.recordAudit(
                            record.id,
                            "date",
                            "inconsistent_value",
                            record.date,
                            normalizedDate,
                            "normalize_date",
                            "Converted supported date representations to ISO format."
                        );

                        record.date = normalizedDate;
                        this.metrics.inconsistentValues++;
                    }
                } catch (error) {
                    this.recordAudit(
                        record.id,
                        "date",
                        "inconsistent_value",
                        record.date,
                        null,
                        "invalidate",
                        error.message
                    );

                    record.date = null;
                }
            }
        }

        this.emit("stageComplete", {
            stage: "normalization",
            records: this.records.length
        });
    }

    removeDuplicates() {
        const seen = new Set();
        const retained = [];

        for (const record of this.records) {
            const key = [
                normalizeText(record.id)?.toLowerCase(),
                normalizeEmail(record.email),
                normalizeDate(record.date)
            ].join("|");

            if (seen.has(key)) {
                this.recordAudit(
                    record.id,
                    "__record__",
                    "duplicate",
                    record,
                    null,
                    "remove_duplicate",
                    "The business identity matches a previously retained transaction."
                );

                this.metrics.duplicatesRemoved++;
                continue;
            }

            seen.add(key);
            retained.push(record);
        }

        this.records = retained;

        this.emit("stageComplete", {
            stage: "duplicate_detection",
            records: this.records.length
        });
    }

    calculateMedian(values) {
        const sorted = [...values].sort((a, b) => a - b);

        if (sorted.length === 0) {
            throw new Error("Cannot calculate median of an empty collection.");
        }

        const middle = Math.floor(sorted.length / 2);

        return sorted.length % 2 === 0
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle];
    }

    handleMissingValues() {
        const validAmounts = this.records
            .map(record => parseAmount(record.amount))
            .filter(value => value !== null);

        if (validAmounts.length === 0) {
            throw new Error("No valid transaction amounts exist for imputation.");
        }

        const amountMedian = this.calculateMedian(validAmounts);

        for (const record of this.records) {
            if (isMissing(record.amount)) {
                record.amount = amountMedian;

                this.recordAudit(
                    record.id,
                    "amount",
                    "missing_value",
                    null,
                    amountMedian,
                    "median_imputation",
                    "A missing numeric amount was replaced with the median."
                );

                this.metrics.missingHandled++;
            }

            // A missing date remains null because a transaction date is not
            // safely inferable from the available fields.
            if (isMissing(record.date)) {
                this.recordAudit(
                    record.id,
                    "date",
                    "missing_value",
                    record.date,
                    null,
                    "retain_missing",
                    "The date cannot be safely reconstructed."
                );

                this.metrics.missingHandled++;
            }
        }

        this.emit("stageComplete", {
            stage: "missing_value_handling",
            records: this.records.length
        });
    }

    detectOutliers() {
        const values = this.records
            .map(record => parseAmount(record.amount))
            .filter(value => value !== null)
            .sort((a, b) => a - b);

        if (values.length < 4) {
            return;
        }

        const middle = Math.floor(values.length / 2);
        const lowerHalf = values.slice(0, middle);
        const upperHalf = values.slice(
            values.length % 2 === 0 ? middle : middle + 1
        );

        const q1 = this.calculateMedian(lowerHalf);
        const q3 = this.calculateMedian(upperHalf);
        const iqr = q3 - q1;

        if (iqr === 0) {
            return;
        }

        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;

        for (const record of this.records) {
            const amount = parseAmount(record.amount);

            if (amount < lowerBound || amount > upperBound) {
                this.recordAudit(
                    record.id,
                    "amount",
                    "outlier",
                    amount,
                    amount,
                    "flag_only",
                    `IQR bounds are ${lowerBound.toFixed(2)} to ${upperBound.toFixed(2)}.`
                );

                this.metrics.outliersFlagged++;
            }
        }

        this.emit("stageComplete", {
            stage: "outlier_detection",
            lowerBound,
            upperBound
        });
    }

    validate() {
        const errors = [];
        const ids = new Set();

        for (const record of this.records) {
            if (!record.id) {
                errors.push("Record has no transaction ID.");
            } else if (ids.has(record.id)) {
                errors.push(`Duplicate transaction ID remains: ${record.id}`);
            } else {
                ids.add(record.id);
            }

            const amount = parseAmount(record.amount);

            if (amount === null || amount < 0) {
                errors.push(`${record.id}: invalid transaction amount.`);
            }

            if (!["India", "United States"].includes(record.country)) {
                errors.push(`${record.id}: unsupported country ${record.country}.`);
            }

            if (!["Electronics", "Home & Kitchen", "Grocery"].includes(record.category)) {
                errors.push(`${record.id}: unsupported category ${record.category}.`);
            }

            if (record.date !== null &&
                !/^\d{4}-\d{2}-\d{2}$/.test(record.date)) {
                errors.push(`${record.id}: invalid canonical date.`);
            }
        }

        this.emit("validation", {
            valid: errors.length === 0,
            errors
        });

        return errors;
    }

    run() {
        this.emit("pipelineStart", {
            records: this.records.length
        });

        this.normalizeValues();
        this.removeDuplicates();
        this.handleMissingValues();
        this.detectOutliers();

        const errors = this.validate();

        if (errors.length > 0) {
            throw new Error(
                `Data validation failed:\n${errors.join("\n")}`
            );
        }

        this.emit("pipelineComplete", {
            records: this.records.length,
            metrics: this.metrics
        });

        return {
            records: this.records,
            audit: this.audit,
            metrics: this.metrics
        };
    }
}

// ---------------------------------------------------------------------------
// CSV persistence
// ---------------------------------------------------------------------------

function csvEscape(value) {
    const text = value === null || value === undefined
        ? ""
        : String(value);

    if (/[",\n]/.test(text)) {
        return `"${text.replace(/"/g, '""')}"`;
    }

    return text;
}

function writeCsv(records, filename) {
    const fields = [
        "id",
        "customer",
        "email",
        "country",
        "state",
        "category",
        "amount",
        "date"
    ];

    const lines = [
        fields.join(","),
        ...records.map(record =>
            fields.map(field => csvEscape(record[field])).join(",")
        )
    ];

    fs.writeFileSync(filename, lines.join("\n"), "utf8");
}

function writeAuditJson(audit, filename) {
    fs.writeFileSync(
        filename,
        JSON.stringify(audit, null, 2),
        "utf8"
    );
}

// ---------------------------------------------------------------------------
// Program execution
// ---------------------------------------------------------------------------

function main() {
    const original = createTransactions();

    console.log("DATA CLEANING WORKFLOW ENGINE");
    console.log("=".repeat(72));
    console.log(`Original records: ${original.length}`);

    const engine = new CleaningEngine(original);

    engine.on("pipelineStart", event => {
        console.log(`Pipeline started with ${event.records} records.`);
    });

    engine.on("stageComplete", event => {
        console.log(
            `Completed ${event.stage}` +
            (event.records !== undefined
                ? `; records=${event.records}`
                : "")
        );
    });

    engine.on("issue", issue => {
        console.log(
            `[${issue.issue}] ${issue.recordId}.${issue.field} -> ${issue.action}`
        );
    });

    engine.on("validation", event => {
        console.log(
            event.valid
                ? "Post-cleaning validation passed."
                : `Validation failed with ${event.errors.length} errors.`
        );
    });

    engine.on("pipelineComplete", event => {
        console.log(
            `Pipeline completed with ${event.records} records.`
        );
        console.log("Metrics:", event.metrics);
    });

    const result = engine.run();

    const outputDirectory = path.join(
        process.cwd(),
        "data_cleaning_output_js"
    );

    fs.mkdirSync(outputDirectory, { recursive: true });

    const cleanedFile = path.join(
        outputDirectory,
        "cleaned_transactions.csv"
    );

    const auditFile = path.join(
        outputDirectory,
        "cleaning_audit.json"
    );

    writeCsv(result.records, cleanedFile);
    writeAuditJson(result.audit, auditFile);

    console.log("\nCLEANED RECORDS");
    console.log("=".repeat(72));

    for (const record of result.records) {
        console.log(
            `${record.id} | ${record.customer} | ` +
            `${record.country} | ${record.state} | ` +
            `${record.category} | ${record.amount.toFixed(2)} | ` +
            `${record.date ?? "MISSING"}`
        );
    }

    console.log("\nAUDIT OUTPUT");
    console.log("=".repeat(72));
    console.log(`Audit entries: ${result.audit.length}`);
    console.log(`CSV: ${cleanedFile}`);
    console.log(`Audit JSON: ${auditFile}`);
}

main();
