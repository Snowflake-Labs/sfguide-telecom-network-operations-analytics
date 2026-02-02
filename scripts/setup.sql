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

-- Create schemas for different network domains
CREATE SCHEMA IF NOT EXISTS RAN_4G COMMENT = '4G Radio Access Network data';
CREATE SCHEMA IF NOT EXISTS RAN_5G COMMENT = '5G Radio Access Network data';
CREATE SCHEMA IF NOT EXISTS CORE_4G COMMENT = '4G Core Network data (EPC)';
CREATE SCHEMA IF NOT EXISTS CORE_5G COMMENT = '5G Core Network data (5GC)';
CREATE SCHEMA IF NOT EXISTS TRANSPORT COMMENT = 'Transport network data (backhaul/fronthaul)';
CREATE SCHEMA IF NOT EXISTS ANALYTICS COMMENT = 'Aggregated analytics and dimension tables';
CREATE SCHEMA IF NOT EXISTS STAGING COMMENT = 'Data staging area';

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

-- Cell Site Dimension
CREATE TABLE IF NOT EXISTS DIM_CELL_SITE (
    CELL_ID VARCHAR(50) PRIMARY KEY,
    SITE_NAME VARCHAR(100),
    REGION VARCHAR(50),
    CITY VARCHAR(100),
    NETWORK_TYPE VARCHAR(20),
    VENDOR VARCHAR(50),
    LOCATION_LAT FLOAT,
    LOCATION_LON FLOAT,
    ANTENNA_HEIGHT_M FLOAT,
    SECTOR_COUNT INTEGER,
    CAPACITY_MBPS FLOAT,
    INSTALL_DATE DATE,
    STATUS VARCHAR(20)
);

-- Network Element Dimension
CREATE TABLE IF NOT EXISTS DIM_NETWORK_ELEMENT (
    ELEMENT_ID VARCHAR(50) PRIMARY KEY,
    ELEMENT_NAME VARCHAR(100),
    ELEMENT_TYPE VARCHAR(50),
    NETWORK_DOMAIN VARCHAR(20),
    VENDOR VARCHAR(50),
    SOFTWARE_VERSION VARCHAR(50),
    CAPACITY FLOAT,
    REGION VARCHAR(50),
    STATUS VARCHAR(20)
);

-- =============================================================================
-- SECTION 8: FACT TABLES
-- =============================================================================

-- RAN Performance Facts (aggregated)
CREATE TABLE IF NOT EXISTS FACT_RAN_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    CELL_ID VARCHAR(50),
    RRC_SUCCESS_RATE FLOAT,
    ERAB_SUCCESS_RATE FLOAT,
    DL_THROUGHPUT_MBPS FLOAT,
    UL_THROUGHPUT_MBPS FLOAT,
    PRB_UTILIZATION FLOAT,
    ACTIVE_USERS INTEGER,
    HANDOVER_SUCCESS_RATE FLOAT,
    LATENCY_MS FLOAT
);

-- Core Performance Facts (aggregated)
CREATE TABLE IF NOT EXISTS FACT_CORE_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    ELEMENT_ID VARCHAR(50),
    ATTACH_SUCCESS_RATE FLOAT,
    PDN_CONN_SUCCESS_RATE FLOAT,
    SERVICE_REQUEST_SUCCESS_RATE FLOAT,
    THROUGHPUT_GBPS FLOAT,
    ACTIVE_BEARERS INTEGER,
    LATENCY_MS FLOAT
);

-- =============================================================================
-- SECTION 9: RAN TABLES (4G)
-- =============================================================================
USE SCHEMA RAN_4G;

CREATE TABLE IF NOT EXISTS ENODEB_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    ENODEB_ID VARCHAR(50),
    CELL_ID VARCHAR(50),
    RRC_CONNESTABATT INTEGER,
    RRC_CONNESTABSUCC INTEGER,
    ERAB_ESTABATT INTEGER,
    ERAB_ESTABSUCC INTEGER,
    DL_PRB_USAGE_RATE FLOAT,
    UL_PRB_USAGE_RATE FLOAT,
    AVG_DL_THPT_MBPS FLOAT,
    AVG_UL_THPT_MBPS FLOAT,
    ACTIVE_UE_COUNT INTEGER,
    HO_OUT_ATT INTEGER,
    HO_OUT_SUCC INTEGER
);

-- =============================================================================
-- SECTION 10: RAN TABLES (5G)
-- =============================================================================
USE SCHEMA RAN_5G;

CREATE TABLE IF NOT EXISTS GNODEB_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    GNODEB_ID VARCHAR(50),
    CELL_ID VARCHAR(50),
    RRC_CONNESTABATT INTEGER,
    RRC_CONNESTABSUCC INTEGER,
    PDU_SESSION_ESTAB_ATT INTEGER,
    PDU_SESSION_ESTAB_SUCC INTEGER,
    DL_PRB_USAGE_RATE FLOAT,
    UL_PRB_USAGE_RATE FLOAT,
    AVG_DL_THPT_MBPS FLOAT,
    AVG_UL_THPT_MBPS FLOAT,
    ACTIVE_UE_COUNT INTEGER,
    XNHO_OUT_ATT INTEGER,
    XNHO_OUT_SUCC INTEGER
);

-- =============================================================================
-- SECTION 11: CORE TABLES (4G)
-- =============================================================================
USE SCHEMA CORE_4G;

CREATE TABLE IF NOT EXISTS MME_4G (
    TIMESTAMP TIMESTAMP_NTZ,
    MME_ID VARCHAR(50),
    ATTACH_ATT INTEGER,
    ATTACH_SUCC INTEGER,
    TAU_ATT INTEGER,
    TAU_SUCC INTEGER,
    SERVICE_REQ_ATT INTEGER,
    SERVICE_REQ_SUCC INTEGER,
    PAGING_ATT INTEGER,
    PAGING_SUCC INTEGER
);

CREATE TABLE IF NOT EXISTS SGW_4G (
    TIMESTAMP TIMESTAMP_NTZ,
    SGW_ID VARCHAR(50),
    CREATE_SESSION_ATT INTEGER,
    CREATE_SESSION_SUCC INTEGER,
    MODIFY_BEARER_ATT INTEGER,
    MODIFY_BEARER_SUCC INTEGER,
    THROUGHPUT_GBPS FLOAT,
    ACTIVE_BEARERS INTEGER
);

CREATE TABLE IF NOT EXISTS PGW_4G (
    TIMESTAMP TIMESTAMP_NTZ,
    PGW_ID VARCHAR(50),
    CREATE_SESSION_ATT INTEGER,
    CREATE_SESSION_SUCC INTEGER,
    IP_ALLOC_ATT INTEGER,
    IP_ALLOC_SUCC INTEGER,
    THROUGHPUT_GBPS FLOAT,
    ACTIVE_PDN_CONN INTEGER
);

-- =============================================================================
-- SECTION 12: CORE TABLES (5G)
-- =============================================================================
USE SCHEMA CORE_5G;

CREATE TABLE IF NOT EXISTS AMF_5G (
    TIMESTAMP TIMESTAMP_NTZ,
    AMF_ID VARCHAR(50),
    REGISTRATION_ATT INTEGER,
    REGISTRATION_SUCC INTEGER,
    DEREGISTRATION_ATT INTEGER,
    DEREGISTRATION_SUCC INTEGER,
    SERVICE_REQ_ATT INTEGER,
    SERVICE_REQ_SUCC INTEGER
);

CREATE TABLE IF NOT EXISTS SMF_5G (
    TIMESTAMP TIMESTAMP_NTZ,
    SMF_ID VARCHAR(50),
    PDU_SESSION_ESTAB_ATT INTEGER,
    PDU_SESSION_ESTAB_SUCC INTEGER,
    PDU_SESSION_MOD_ATT INTEGER,
    PDU_SESSION_MOD_SUCC INTEGER,
    ACTIVE_SESSIONS INTEGER
);

CREATE TABLE IF NOT EXISTS UPF_5G (
    TIMESTAMP TIMESTAMP_NTZ,
    UPF_ID VARCHAR(50),
    THROUGHPUT_GBPS FLOAT,
    PACKET_DROP_RATE FLOAT,
    LATENCY_MS FLOAT,
    ACTIVE_SESSIONS INTEGER
);

-- =============================================================================
-- SECTION 13: TRANSPORT TABLES
-- =============================================================================
USE SCHEMA TRANSPORT;

CREATE TABLE IF NOT EXISTS TRANSPORT_DEVICE_PERFORMANCE (
    TIMESTAMP TIMESTAMP_NTZ,
    DEVICE_ID VARCHAR(50),
    DEVICE_TYPE VARCHAR(50),
    LINK_UTILIZATION FLOAT,
    LATENCY_MS FLOAT,
    PACKET_LOSS_RATE FLOAT,
    JITTER_MS FLOAT,
    THROUGHPUT_GBPS FLOAT
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
import csv
from io import StringIO
from snowflake.snowpark import Session

def load_data(session: Session) -> str:
    base_url = "https://raw.githubusercontent.com/Snowflake-Labs/sfguide-telecom-network-operations-analytics/main/scripts/csvs"
    
    files_to_load = [
        ("DIM_CELL_SITE.csv", "NETWORK_OPERATIONS.ANALYTICS.DIM_CELL_SITE"),
        ("DIM_NETWORK_ELEMENT.csv", "NETWORK_OPERATIONS.ANALYTICS.DIM_NETWORK_ELEMENT"),
        ("ENODEB_PERFORMANCE.csv", "NETWORK_OPERATIONS.RAN_4G.ENODEB_PERFORMANCE"),
        ("GNODEB_PERFORMANCE.csv", "NETWORK_OPERATIONS.RAN_5G.GNODEB_PERFORMANCE"),
        ("MME_4G.csv", "NETWORK_OPERATIONS.CORE_4G.MME_4G"),
        ("SGW_4G.csv", "NETWORK_OPERATIONS.CORE_4G.SGW_4G"),
        ("PGW_4G.csv", "NETWORK_OPERATIONS.CORE_4G.PGW_4G"),
        ("AMF_5G.csv", "NETWORK_OPERATIONS.CORE_5G.AMF_5G"),
        ("SMF_5G.csv", "NETWORK_OPERATIONS.CORE_5G.SMF_5G"),
        ("UPF_5G.csv", "NETWORK_OPERATIONS.CORE_5G.UPF_5G"),
        ("TRANSPORT_DEVICE_PERFORMANCE.csv", "NETWORK_OPERATIONS.TRANSPORT.TRANSPORT_DEVICE_PERFORMANCE"),
        ("FACT_RAN_PERFORMANCE.csv", "NETWORK_OPERATIONS.ANALYTICS.FACT_RAN_PERFORMANCE"),
        ("FACT_CORE_PERFORMANCE.csv", "NETWORK_OPERATIONS.ANALYTICS.FACT_CORE_PERFORMANCE"),
    ]
    
    results = []
    
    for filename, table_name in files_to_load:
        try:
            url = f"{base_url}/{filename}"
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            csv_content = StringIO(response.text)
            reader = csv.reader(csv_content)
            header = next(reader)
            
            rows = list(reader)
            
            if rows:
                df = session.create_dataframe(rows, schema=header)
                df.write.mode("overwrite").save_as_table(table_name)
                results.append(f"✓ {filename}: {len(rows)} rows loaded into {table_name}")
            else:
                results.append(f"⚠ {filename}: No data found")
                
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
    SITES as DIM_CELL_SITE primary key (CELL_SITE_ID) with synonyms=('cell sites','towers','cells','base stations') comment='Cell site locations and information',
    PERFORMANCE as FACT_RAN_PERFORMANCE primary key (TIMESTAMP, CELL_SITE_ID) with synonyms=('network performance','ran metrics','kpis') comment='Network performance metrics'
  )
  relationships (
    PERF_TO_SITES as PERFORMANCE(CELL_SITE_ID) references SITES(CELL_SITE_ID)
  )
  facts (
    PERFORMANCE.RRC_SUCCESS_RATE as RRC_Success comment='RRC success rate %',
    PERFORMANCE.ERAB_SUCCESS_RATE as ERAB_Success comment='E-RAB success rate %',
    PERFORMANCE.PRB_UTILIZATION as PRB_Util comment='PRB utilization %',
    PERFORMANCE.DL_THROUGHPUT_MBPS as DL_Throughput comment='DL throughput Mbps',
    PERFORMANCE.UL_THROUGHPUT_MBPS as UL_Throughput comment='UL throughput Mbps',
    PERFORMANCE.HANDOVER_SUCCESS_RATE as HO_Success comment='Handover success rate %',
    PERFORMANCE.ACTIVE_USERS as Active_Users comment='Active users count',
    PERFORMANCE.AVAILABILITY as Availability comment='Cell availability %'
  )
  dimensions (
    PERFORMANCE.TIMESTAMP as Timestamp with synonyms=('time','when') comment='Measurement time',
    SITES.CELL_SITE_ID as Cell_ID comment='Cell site ID',
    SITES.CELL_SITE_NAME as Cell_Name comment='Cell site name',
    SITES.CITY as City with synonyms=('city','location') comment='City',
    SITES.REGION as Region with synonyms=('region','area') comment='Region',
    SITES.TECHNOLOGY as Technology with synonyms=('tech','4g','5g') comment='Technology type'
  )
  metrics (
    AVG_RRC_SUCCESS as AVG(PERFORMANCE.RRC_SUCCESS_RATE) comment='Average RRC success rate % - target >= 95%',
    AVG_PRB_UTIL as AVG(PERFORMANCE.PRB_UTILIZATION) comment='Average PRB utilization % - warning > 70%',
    AVG_DL_THROUGHPUT as AVG(PERFORMANCE.DL_THROUGHPUT_MBPS) comment='Average DL throughput Mbps',
    AVG_UL_THROUGHPUT as AVG(PERFORMANCE.UL_THROUGHPUT_MBPS) comment='Average UL throughput Mbps',
    AVG_AVAILABILITY as AVG(PERFORMANCE.AVAILABILITY) comment='Average availability % - target >= 99%',
    TOTAL_SITES as COUNT(DISTINCT SITES.CELL_SITE_ID) comment='Total cell sites',
    AVG_HO_SUCCESS as AVG(PERFORMANCE.HANDOVER_SUCCESS_RATE) comment='Average handover success rate %'
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
    "orchestration": "Data from Sept 2025. Use MAX(Timestamp) then DATEADD backwards for ranges. Default 24h. Join facts to sites for geography. Portugal has 5 cities, 5 regions. Technologies: 4G and 5G. PRB >70%=warning, >85%=critical.",
    "sample_questions": [
      {"question":"Which cells have RRC below 95%?"},
      {"question":"Show top 10 congested sites"},
      {"question":"Compare 4G vs 5G throughput"},
      {"question":"What is average PRB by region?"},
      {"question":"Show Lisboa performance last 24h"},
      {"question":"Which sites have availability issues?"}
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
