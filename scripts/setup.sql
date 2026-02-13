-- Copyright 2026 Snowflake Inc.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

/*******************************************************************************
 * Network Operations Reporting and Analytics with Snowflake Intelligence
 * Setup Script
 * 
 * This script creates all necessary Snowflake objects for the Network Operations
 * solution including database, schemas, tables, stages, and warehouses.
 * 
 * Data is loaded automatically from GitHub using External Access Integration.
 ******************************************************************************/

-- =============================================================================
-- SECTION 1: WAREHOUSES
-- =============================================================================
USE ROLE ACCOUNTADMIN;

CREATE WAREHOUSE IF NOT EXISTS NETWORK_OPS_WH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for Network Operations queries and dashboard';

CREATE WAREHOUSE IF NOT EXISTS NETWORK_OPS_BUILD_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for data loading operations';

-- =============================================================================
-- SECTION 2: ROLE SETUP
-- =============================================================================
CREATE ROLE IF NOT EXISTS NETWORK_OPS_ANALYST
    COMMENT = 'Role for Network Operations Analytics users';

SET my_user_var = (SELECT '"' || CURRENT_USER() || '"');
GRANT ROLE NETWORK_OPS_ANALYST TO USER identifier($my_user_var);

GRANT USAGE ON WAREHOUSE NETWORK_OPS_WH TO ROLE NETWORK_OPS_ANALYST;
GRANT USAGE ON WAREHOUSE NETWORK_OPS_BUILD_WH TO ROLE NETWORK_OPS_ANALYST;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE NETWORK_OPS_ANALYST;

-- =============================================================================
-- SECTION 3: DATABASE AND SCHEMAS
-- =============================================================================
CREATE DATABASE IF NOT EXISTS NETWORK_OPERATIONS
    COMMENT = 'Network Operations Reporting and Analytics Database';

USE DATABASE NETWORK_OPERATIONS;

-- Grant all privileges to ACCOUNTADMIN to ensure schema creation works
GRANT ALL PRIVILEGES ON DATABASE NETWORK_OPERATIONS TO ROLE ACCOUNTADMIN;

-- Create schemas for different network domains
CREATE SCHEMA IF NOT EXISTS RAN_4G COMMENT = '4G Radio Access Network data';
CREATE SCHEMA IF NOT EXISTS RAN_5G COMMENT = '5G Radio Access Network data';
CREATE SCHEMA IF NOT EXISTS CORE_4G COMMENT = '4G Core Network data (EPC)';
CREATE SCHEMA IF NOT EXISTS CORE_5G COMMENT = '5G Core Network data (5GC)';
CREATE SCHEMA IF NOT EXISTS TRANSPORT COMMENT = 'Transport network data (backhaul/fronthaul)';
CREATE SCHEMA IF NOT EXISTS ANALYTICS COMMENT = 'Aggregated analytics and dimension tables';
CREATE SCHEMA IF NOT EXISTS STAGING COMMENT = 'Data staging area';

-- Grant ownership to analyst role after all objects created
GRANT OWNERSHIP ON DATABASE NETWORK_OPERATIONS TO ROLE NETWORK_OPS_ANALYST COPY CURRENT GRANTS;

-- =============================================================================
-- SECTION 4: GITHUB EXTERNAL ACCESS INTEGRATION
-- =============================================================================
USE SCHEMA STAGING;

CREATE OR REPLACE NETWORK RULE GITHUB_NETWORK_RULE
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('raw.githubusercontent.com');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION GITHUB_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (GITHUB_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'External access to GitHub for loading CSV data files';

-- =============================================================================
-- SECTION 5: STAGES
-- =============================================================================
CREATE STAGE IF NOT EXISTS CSV_DATA
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for CSV data files';

CREATE STAGE IF NOT EXISTS STREAMLIT_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Streamlit application files';

-- =============================================================================
-- SECTION 6: FILE FORMAT
-- =============================================================================
CREATE FILE FORMAT IF NOT EXISTS CSV_FORMAT
    TYPE = 'CSV'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('NULL', 'null', '')
    EMPTY_FIELD_AS_NULL = TRUE;

-- =============================================================================
-- SECTION 7: DIMENSION TABLES
-- =============================================================================
USE SCHEMA ANALYTICS;

-- Cell Site Dimension (matches DIM_CELL_SITE.csv)
CREATE TABLE IF NOT EXISTS DIM_CELL_SITE (
    CELL_ID VARCHAR(50) PRIMARY KEY,
    SITE_ID VARCHAR(50),
    NODE_ID VARCHAR(50),
    TECHNOLOGY VARCHAR(20),
    LOCATION_LAT FLOAT,
    LOCATION_LON FLOAT,
    TRANSPORT_DEVICE_ID VARCHAR(50),
    TRANSPORT_INTERFACE_ID VARCHAR(50),
    REGION VARCHAR(50),
    CITY VARCHAR(100),
    INSTALLATION_DATE DATE,
    STATUS VARCHAR(20)
);

-- Network Element Dimension (matches DIM_NETWORK_ELEMENT.csv)
CREATE TABLE IF NOT EXISTS DIM_NETWORK_ELEMENT (
    ELEMENT_ID VARCHAR(50) PRIMARY KEY,
    ELEMENT_TYPE VARCHAR(50),
    ELEMENT_SUBTYPE VARCHAR(50),
    VENDOR VARCHAR(50),
    SOFTWARE_VERSION VARCHAR(50),
    HARDWARE_MODEL VARCHAR(100),
    LOCATION_LAT FLOAT,
    LOCATION_LON FLOAT,
    REGION VARCHAR(50),
    INSTALLATION_DATE DATE,
    CAPACITY_RATING FLOAT,
    STATUS VARCHAR(20)
);

-- =============================================================================
-- SECTION 8: FACT TABLES
-- =============================================================================

-- RAN Performance Facts (matches FACT_RAN_PERFORMANCE.csv)
CREATE TABLE IF NOT EXISTS FACT_RAN_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    CELL_ID VARCHAR(50),
    TECHNOLOGY VARCHAR(20),
    RRC_CONNESTABATT INTEGER,
    RRC_CONNESTABSUCC INTEGER,
    DL_PRB_UTILIZATION FLOAT,
    UL_PRB_UTILIZATION FLOAT,
    CELL_AVAILABILITY FLOAT,
    DL_THROUGHPUT_MBPS FLOAT,
    UL_THROUGHPUT_MBPS FLOAT,
    HANDOVER_ATTEMPTS INTEGER,
    HANDOVER_SUCCESSES INTEGER,
    ERAB_ESTABSUCCRATE FLOAT,
    PDU_SESSESTABSUCCRATE FLOAT
);

-- Core Performance Facts (matches FACT_CORE_PERFORMANCE.csv)
CREATE TABLE IF NOT EXISTS FACT_CORE_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    NODE_ID VARCHAR(50),
    NODE_TYPE VARCHAR(50),
    TECHNOLOGY VARCHAR(20),
    ACTIVE_SESSIONS INTEGER,
    CPU_LOAD FLOAT,
    MEMORY_UTILIZATION FLOAT,
    REGISTRATION_ATTEMPTS INTEGER,
    REGISTRATION_SUCCESSES INTEGER,
    SESSION_ESTABLISH_ATTEMPTS INTEGER,
    SESSION_ESTABLISH_SUCCESSES INTEGER,
    PAGING_ATTEMPTS INTEGER,
    DL_DATA_VOLUME_GB FLOAT,
    UL_DATA_VOLUME_GB FLOAT
);

-- =============================================================================
-- SECTION 9: RAN TABLES (4G)
-- =============================================================================
USE SCHEMA RAN_4G;

-- Matches ENODEB_PERFORMANCE.csv
CREATE TABLE IF NOT EXISTS ENODEB_PERFORMANCE (
    ENODEB_ID VARCHAR(50),
    VENDOR VARCHAR(50),
    CELL_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    RRC_CONNESTABATT INTEGER,
    RRC_CONNESTABSUCC INTEGER,
    S1SIG_CONNESTABATT INTEGER,
    S1SIG_CONNESTABSUCC INTEGER,
    ERAB_ESTABINITATTNBR_QCI INTEGER,
    ERAB_ESTABINITsuccnbr_QCI INTEGER,
    X2_HO_EXECSUCCNBR INTEGER,
    DL_PRB_UTILIZATION FLOAT,
    UL_PRB_UTILIZATION FLOAT,
    DL_THROUGHPUT_CELL FLOAT,
    UL_THROUGHPUT_CELL FLOAT,
    CELL_AVAILABILITY FLOAT,
    F1AP_PAGINGRECEIVEDNBR INTEGER
);

-- =============================================================================
-- SECTION 10: RAN TABLES (5G)
-- =============================================================================
USE SCHEMA RAN_5G;

-- Matches GNODEB_PERFORMANCE.csv
CREATE TABLE IF NOT EXISTS GNODEB_PERFORMANCE (
    GNODEB_ID VARCHAR(50),
    VENDOR VARCHAR(50),
    CELL_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    RRC_CONNESTABATT INTEGER,
    RRC_CONNESTABSUCC INTEGER,
    NGAP_PDUSESSRESOURCESETUPATT INTEGER,
    NGAP_PDUSESSRESOURCESETUPSUCC INTEGER,
    QOS_FLOWS_ESTAB_SUCC INTEGER,
    ERAB_ESTABINITSUCC_NBR_QCI INTEGER,
    XN_HO_EXECSUCCNBR INTEGER,
    RRU_PRB_USED_DL FLOAT,
    RRU_PRB_USED_UL FLOAT,
    DRB_UETHPDL FLOAT,
    DRB_UETHPUL FLOAT,
    CELL_AVAILABILITY FLOAT,
    F1AP_PAGINGRECEIVEDNBR INTEGER
);

-- =============================================================================
-- SECTION 11: CORE TABLES (4G)
-- =============================================================================
USE SCHEMA CORE_4G;

-- Matches MME_4G.csv
CREATE TABLE IF NOT EXISTS MME_4G (
    MME_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    VENDOR VARCHAR(50),
    MM_ATTACHEDUES INTEGER,
    MM_ATTACHATT INTEGER,
    MM_ATTACHSUCC INTEGER,
    CPU_LOAD FLOAT,
    PAGING_ATTEMPTS INTEGER,
    MM_TAU_ATT INTEGER,
    MM_TAU_SUCC INTEGER,
    S6A_AUTHINFOREQ INTEGER,
    S6A_AUTHINFOSUCC INTEGER
);

-- Matches SGW_4G.csv
CREATE TABLE IF NOT EXISTS SGW_4G (
    SGW_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    GTP_ACTIVETUNNELS INTEGER,
    GTP_DL_THROUGHPUT FLOAT,
    GTP_UL_THROUGHPUT FLOAT,
    S1U_VOLUME_DL FLOAT,
    S1U_VOLUME_UL FLOAT,
    S5S8_VOLUME_DL FLOAT
);

-- Matches PGW_4G.csv
CREATE TABLE IF NOT EXISTS PGW_4G (
    PGW_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    SM_ACTIVEPDNSESSIONS INTEGER,
    SM_PDNCONNESTABATT INTEGER,
    SM_PDNCONNESTABSUCC INTEGER,
    SGI_VOLUME_DL FLOAT,
    SGI_VOLUME_UL FLOAT,
    APN_DATA_VOLUME FLOAT
);

-- =============================================================================
-- SECTION 12: CORE TABLES (5G)
-- =============================================================================
USE SCHEMA CORE_5G;

-- Matches AMF_5G.csv
CREATE TABLE IF NOT EXISTS AMF_5G (
    AMF_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    VENDOR VARCHAR(50),
    RM_REGISTEREDUES INTEGER,
    RM_REGATT INTEGER,
    RM_REGSUCC INTEGER,
    CPU_LOAD FLOAT,
    PAGING_ATTEMPTS INTEGER,
    MM_MOBILITYREGUPDATEATT INTEGER,
    MM_MOBILITYREGUPDATESUCC INTEGER
);

-- Matches SMF_5G.csv
CREATE TABLE IF NOT EXISTS SMF_5G (
    SMF_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    SM_PDUSESSESTABATT INTEGER,
    SM_PDUSESSESTABSUCC INTEGER,
    SM_ACTIVEPDUSESSIONS INTEGER,
    N4_SESSESTABREQ INTEGER,
    N4_SESSESTABSUCC INTEGER
);

-- Matches UPF_5G.csv
CREATE TABLE IF NOT EXISTS UPF_5G (
    UPF_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    GTP_N3_OCTETS_DL FLOAT,
    GTP_N3_OCTETS_UL FLOAT,
    GTP_N3_PACKETS_DL FLOAT,
    GTP_N3_PACKETS_UL FLOAT,
    N6_THROUGHPUT_DL FLOAT,
    N6_THROUGHPUT_UL FLOAT,
    PACKET_LOSS_RATE_UL FLOAT
);

-- =============================================================================
-- SECTION 13: TRANSPORT TABLES
-- =============================================================================
USE SCHEMA TRANSPORT;

-- Matches TRANSPORT_DEVICE_PERFORMANCE.csv
CREATE TABLE IF NOT EXISTS TRANSPORT_DEVICE_PERFORMANCE (
    DEVICE_ID VARCHAR(50),
    INTERFACE_ID VARCHAR(50),
    TIMESTAMP TIMESTAMP_NTZ,
    BANDWIDTH_UTILIZATION_IN FLOAT,
    BANDWIDTH_UTILIZATION_OUT FLOAT,
    INTERFACE_DISCARDS_IN INTEGER,
    INTERFACE_ERRORS_OUT INTEGER,
    LATENCY_MS FLOAT,
    JITTER_MS FLOAT,
    PACKET_LOSS_PERCENT FLOAT,
    CPU_LOAD FLOAT
);

-- =============================================================================
-- SECTION 14: GITHUB DATA LOADER PROCEDURE
-- =============================================================================
USE SCHEMA STAGING;

CREATE OR REPLACE PROCEDURE LOAD_DATA_FROM_GITHUB()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'requests')
HANDLER = 'load_data'
EXTERNAL_ACCESS_INTEGRATIONS = (GITHUB_ACCESS_INTEGRATION)
AS
$$
import requests
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from snowflake.snowpark import Session

def load_data(session: Session) -> str:
    base_url = "https://raw.githubusercontent.com/Snowflake-Labs/sfguide-telecom-network-operations-analytics/main/scripts/csvs"
    stage_path = "@NETWORK_OPERATIONS.STAGING.CSV_DATA"
    
    files_to_load = [
        ("DIM_CELL_SITE.csv.gz", "NETWORK_OPERATIONS.ANALYTICS.DIM_CELL_SITE"),
        ("DIM_NETWORK_ELEMENT.csv.gz", "NETWORK_OPERATIONS.ANALYTICS.DIM_NETWORK_ELEMENT"),
        ("ENODEB_PERFORMANCE.csv.gz", "NETWORK_OPERATIONS.RAN_4G.ENODEB_PERFORMANCE"),
        ("GNODEB_PERFORMANCE.csv.gz", "NETWORK_OPERATIONS.RAN_5G.GNODEB_PERFORMANCE"),
        ("MME_4G.csv.gz", "NETWORK_OPERATIONS.CORE_4G.MME_4G"),
        ("SGW_4G.csv.gz", "NETWORK_OPERATIONS.CORE_4G.SGW_4G"),
        ("PGW_4G.csv.gz", "NETWORK_OPERATIONS.CORE_4G.PGW_4G"),
        ("AMF_5G.csv.gz", "NETWORK_OPERATIONS.CORE_5G.AMF_5G"),
        ("SMF_5G.csv.gz", "NETWORK_OPERATIONS.CORE_5G.SMF_5G"),
        ("UPF_5G.csv.gz", "NETWORK_OPERATIONS.CORE_5G.UPF_5G"),
        ("TRANSPORT_DEVICE_PERFORMANCE.csv.gz", "NETWORK_OPERATIONS.TRANSPORT.TRANSPORT_DEVICE_PERFORMANCE"),
        ("FACT_RAN_PERFORMANCE.csv.gz", "NETWORK_OPERATIONS.ANALYTICS.FACT_RAN_PERFORMANCE"),
        ("FACT_CORE_PERFORMANCE.csv.gz", "NETWORK_OPERATIONS.ANALYTICS.FACT_CORE_PERFORMANCE"),
    ]
    
    downloaded_files = {}
    
    def download_file(filename):
        url = f"{base_url}/{filename}"
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        local_path = f"/tmp/{filename}"
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return filename, local_path
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_file, f[0]): f for f in files_to_load}
        for future in as_completed(futures):
            try:
                filename, local_path = future.result()
                downloaded_files[filename] = local_path
            except Exception as e:
                filename = futures[future][0]
                downloaded_files[filename] = f"ERROR: {str(e)}"
    
    results = []
    for filename, table_name in files_to_load:
        try:
            if filename not in downloaded_files or downloaded_files[filename].startswith("ERROR"):
                results.append(f"✗ {filename}: Download failed - {downloaded_files.get(filename, 'Unknown error')}")
                continue
            
            local_path = downloaded_files[filename]
            session.file.put(local_path, stage_path, auto_compress=False, overwrite=True)
            
            copy_sql = f"""
                COPY INTO {table_name}
                FROM {stage_path}/{filename}
                FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' NULL_IF = ('NULL', 'null', '') COMPRESSION = 'GZIP')
                PURGE = FALSE
                ON_ERROR = 'CONTINUE'
            """
            session.sql(f"TRUNCATE TABLE IF EXISTS {table_name}").collect()
            result = session.sql(copy_sql).collect()
            rows_loaded = result[0]['rows_loaded'] if result else 0
            results.append(f"✓ {filename}: {rows_loaded} rows loaded into {table_name}")
                
        except Exception as e:
            results.append(f"✗ {filename}: Error - {str(e)}")
    
    return "\n".join(results)
$$;

-- =============================================================================
-- SECTION 15: STREAMLIT APP LOADER PROCEDURE
-- =============================================================================
CREATE OR REPLACE PROCEDURE LOAD_STREAMLIT_FROM_GITHUB()
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'requests')
HANDLER = 'load_streamlit'
EXTERNAL_ACCESS_INTEGRATIONS = (GITHUB_ACCESS_INTEGRATION)
AS
$$
import requests
from snowflake.snowpark import Session
from snowflake.snowpark.files import SnowflakeFile

def load_streamlit(session: Session) -> str:
    base_url = "https://raw.githubusercontent.com/Snowflake-Labs/sfguide-telecom-network-operations-analytics/main/scripts/streamlit"
    stage_path = "@NETWORK_OPERATIONS.STAGING.STREAMLIT_STAGE"
    
    files_to_load = ["streamlit_app.py", "environment.yml"]
    results = []
    
    for filename in files_to_load:
        try:
            url = f"{base_url}/{filename}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            local_path = f"/tmp/{filename}"
            with open(local_path, 'w') as f:
                f.write(response.text)
            
            session.file.put(local_path, stage_path, auto_compress=False, overwrite=True)
            results.append(f"✓ {filename}: uploaded to {stage_path}")
            
        except Exception as e:
            results.append(f"✗ {filename}: Error - {str(e)}")
    
    return "\n".join(results)
$$;

-- =============================================================================
-- SECTION 16: LOAD DATA AND STREAMLIT FROM GITHUB
-- =============================================================================
USE WAREHOUSE NETWORK_OPS_BUILD_WH;

CALL STAGING.LOAD_DATA_FROM_GITHUB();
CALL STAGING.LOAD_STREAMLIT_FROM_GITHUB();

-- =============================================================================
-- SECTION 17: VERIFY DATA LOAD
-- =============================================================================
SELECT 'DIM_CELL_SITE' AS TABLE_NAME, COUNT(*) AS RECORD_COUNT FROM ANALYTICS.DIM_CELL_SITE
UNION ALL SELECT 'DIM_NETWORK_ELEMENT', COUNT(*) FROM ANALYTICS.DIM_NETWORK_ELEMENT
UNION ALL SELECT 'ENODEB_PERFORMANCE', COUNT(*) FROM RAN_4G.ENODEB_PERFORMANCE
UNION ALL SELECT 'GNODEB_PERFORMANCE', COUNT(*) FROM RAN_5G.GNODEB_PERFORMANCE
UNION ALL SELECT 'FACT_RAN_PERFORMANCE', COUNT(*) FROM ANALYTICS.FACT_RAN_PERFORMANCE
UNION ALL SELECT 'FACT_CORE_PERFORMANCE', COUNT(*) FROM ANALYTICS.FACT_CORE_PERFORMANCE
UNION ALL SELECT 'TRANSPORT_DEVICE_PERFORMANCE', COUNT(*) FROM TRANSPORT.TRANSPORT_DEVICE_PERFORMANCE
ORDER BY TABLE_NAME;

-- =============================================================================
-- SECTION 18: CREATE STREAMLIT APPLICATION
-- =============================================================================
USE SCHEMA ANALYTICS;

CREATE OR REPLACE STREAMLIT NETWORK_OPS_DASHBOARD
    ROOT_LOCATION = '@NETWORK_OPERATIONS.STAGING.STREAMLIT_STAGE'
    MAIN_FILE = 'streamlit_app.py'
    QUERY_WAREHOUSE = 'NETWORK_OPS_WH'
    COMMENT = '{"origin":"sf_sit-is", "name":"network_operations", "version":{"major":1, "minor":0}}';

-- =============================================================================
-- SECTION 19: GRANT STREAMLIT ACCESS TO ROLE
-- =============================================================================
GRANT USAGE ON SCHEMA NETWORK_OPERATIONS.ANALYTICS TO ROLE NETWORK_OPS_ANALYST;
GRANT USAGE ON STREAMLIT NETWORK_OPERATIONS.ANALYTICS.NETWORK_OPS_DASHBOARD TO ROLE NETWORK_OPS_ANALYST;

-- =============================================================================
-- SECTION 20: CREATE SEMANTIC VIEW
-- =============================================================================
USE SCHEMA ANALYTICS;

CREATE DATABASE IF NOT EXISTS SNOWFLAKE_INTELLIGENCE;
CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS;

CREATE OR REPLACE SEMANTIC VIEW NETWORK_OPERATIONS.ANALYTICS.NETWORK_SEMANTIC_VIEW
  tables (
    SITES as DIM_CELL_SITE primary key (CELL_ID) with synonyms=('cell sites','towers','cells','base stations') comment='Cell site locations and information',
    PERFORMANCE as FACT_RAN_PERFORMANCE primary key (TIMESTAMP, CELL_ID) with synonyms=('network performance','ran metrics','kpis') comment='Network performance metrics'
  )
  relationships (
    PERF_TO_SITES as PERFORMANCE(CELL_ID) references SITES(CELL_ID)
  )
  facts (
    PERFORMANCE.RRC_CONNESTABATT as RRC_Attempts comment='RRC connection establishment attempts',
    PERFORMANCE.RRC_CONNESTABSUCC as RRC_Successes comment='RRC connection establishment successes',
    PERFORMANCE.ERAB_ESTABSUCCRATE as ERAB_Success_Rate comment='E-RAB establishment success rate %',
    PERFORMANCE.DL_PRB_UTILIZATION as DL_PRB_Util comment='DL PRB utilization %',
    PERFORMANCE.UL_PRB_UTILIZATION as UL_PRB_Util comment='UL PRB utilization %',
    PERFORMANCE.CELL_AVAILABILITY as Cell_Availability comment='Cell availability %',
    PERFORMANCE.DL_THROUGHPUT_MBPS as DL_Throughput comment='DL throughput Mbps',
    PERFORMANCE.UL_THROUGHPUT_MBPS as UL_Throughput comment='UL throughput Mbps',
    PERFORMANCE.HANDOVER_ATTEMPTS as HO_Attempts comment='Handover attempts',
    PERFORMANCE.HANDOVER_SUCCESSES as HO_Successes comment='Handover successes'
  )
  dimensions (
    PERFORMANCE.TIMESTAMP as Timestamp with synonyms=('time','when') comment='Measurement time',
    PERFORMANCE.TECHNOLOGY as Technology with synonyms=('tech','4g','5g') comment='Technology type (4G/5G)',
    SITES.CELL_ID as Cell_ID comment='Cell site ID',
    SITES.SITE_ID as Site_ID comment='Site identifier',
    SITES.CITY as City with synonyms=('city','location') comment='City',
    SITES.REGION as Region with synonyms=('region','area') comment='Region'
  )
  metrics (
    RRC_SUCCESS_RATE as (SUM(PERFORMANCE.RRC_CONNESTABSUCC) * 100.0 / NULLIF(SUM(PERFORMANCE.RRC_CONNESTABATT), 0)) comment='RRC success rate % - target >= 95%',
    AVG_DL_PRB_UTIL as AVG(PERFORMANCE.DL_PRB_UTILIZATION) comment='Average DL PRB utilization % - warning > 70%',
    AVG_UL_PRB_UTIL as AVG(PERFORMANCE.UL_PRB_UTILIZATION) comment='Average UL PRB utilization %',
    AVG_DL_THROUGHPUT as AVG(PERFORMANCE.DL_THROUGHPUT_MBPS) comment='Average DL throughput Mbps',
    AVG_UL_THROUGHPUT as AVG(PERFORMANCE.UL_THROUGHPUT_MBPS) comment='Average UL throughput Mbps',
    AVG_AVAILABILITY as AVG(PERFORMANCE.CELL_AVAILABILITY) comment='Average cell availability % - target >= 99%',
    TOTAL_SITES as COUNT(DISTINCT SITES.CELL_ID) comment='Total cell sites',
    HO_SUCCESS_RATE as (SUM(PERFORMANCE.HANDOVER_SUCCESSES) * 100.0 / NULLIF(SUM(PERFORMANCE.HANDOVER_ATTEMPTS), 0)) comment='Handover success rate %'
  )
  comment='Network performance semantic view for natural language queries';

-- Verify semantic view
DESCRIBE SEMANTIC VIEW NETWORK_SEMANTIC_VIEW;

-- =============================================================================
-- SECTION 21: CREATE SNOWFLAKE INTELLIGENCE AGENT
-- =============================================================================

GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
GRANT SELECT ON ALL SEMANTIC VIEWS IN SCHEMA NETWORK_OPERATIONS.ANALYTICS TO ROLE ACCOUNTADMIN;

CREATE OR REPLACE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.NETWORK_OPERATIONS_AGENT
WITH PROFILE='{"display_name":"Network Operations AI Agent"}'
COMMENT='AI agent for network performance analysis - ask questions about 4G/5G network performance, RRC success, throughput, capacity, and regional analysis in Portugal'
FROM SPECIFICATION $$
{
  "models": {"orchestration": ""},
  "instructions": {
    "response": "You are a telecom network analyst for Portugal network. Provide insights with charts. Use bar charts for comparisons, line charts for trends. Thresholds: RRC >=95%, PRB <70%, Availability >=99%, Throughput >=10 Mbps.",
    "orchestration": "Data from Sept 2025. Use MAX(Timestamp) then DATEADD backwards for ranges. Default 24h. Join facts to sites for geography. Portugal has 5 cities, 5 regions. Technologies: 4G and 5G. PRB >70%=warning, >85%=critical. RRC success rate = RRC_CONNESTABSUCC/RRC_CONNESTABATT*100.",
    "sample_questions": [
      {"question":"Which cells have RRC success rate below 95%?"},
      {"question":"Show top 10 congested sites by PRB utilization"},
      {"question":"Compare 4G vs 5G throughput"},
      {"question":"What is average PRB utilization by region?"},
      {"question":"Show Lisboa performance last 24h"},
      {"question":"Which sites have availability below 99%?"}
    ]
  },
  "tools": [{
    "tool_spec": {
      "type": "cortex_analyst_text_to_sql",
      "name": "Query Network Data",
      "description": "Query network performance: RRC success, throughput, PRB utilization, availability, handover success across 450 sites in Portugal"
    }
  }],
  "tool_resources": {
    "Query Network Data": {
      "semantic_view": "NETWORK_OPERATIONS.ANALYTICS.NETWORK_SEMANTIC_VIEW",
      "execution_environment": {
        "type": "warehouse",
        "warehouse": "NETWORK_OPS_WH"
      }
    }
  }
}
$$;

-- Grant analyst role access to the agent
GRANT USAGE ON AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.NETWORK_OPERATIONS_AGENT TO ROLE NETWORK_OPS_ANALYST;
GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE NETWORK_OPS_ANALYST;

SHOW AGENTS IN SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS;
-- =============================================================================

/*
USE ROLE ACCOUNTADMIN;

-- Drop Agent (if created)
DROP AGENT IF EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS.NETWORK_OPERATIONS_AGENT;

-- Drop Streamlit (if created)
DROP STREAMLIT IF EXISTS NETWORK_OPERATIONS.ANALYTICS.NETWORK_OPS_DASHBOARD;

-- Drop External Access Integration
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS GITHUB_ACCESS_INTEGRATION;
DROP NETWORK RULE IF EXISTS NETWORK_OPERATIONS.STAGING.GITHUB_NETWORK_RULE;

-- Drop Database (cascades to all schemas, tables, stages)
DROP DATABASE IF EXISTS NETWORK_OPERATIONS;

-- Drop Warehouses
DROP WAREHOUSE IF EXISTS NETWORK_OPS_WH;
DROP WAREHOUSE IF EXISTS NETWORK_OPS_BUILD_WH;

-- Drop Role
DROP ROLE IF EXISTS NETWORK_OPS_ANALYST;
*/
