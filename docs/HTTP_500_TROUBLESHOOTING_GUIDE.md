# Comprehensive HTTP 500 Internal Server Error Troubleshooting Guide

**Last Updated:** December 2025
**Target Audience:** Developers, System Administrators, DevOps Engineers
**Severity Level:** CRITICAL - Production Impact

---

## Table of Contents

1. [Quick Reference Emergency Checklist](#quick-reference-emergency-checklist)
2. [Understanding HTTP 500 Errors](#understanding-http-500-errors)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Systematic Debugging Process](#systematic-debugging-process)
5. [Language-Specific Debugging](#language-specific-debugging)
6. [Web Server-Specific Solutions](#web-server-specific-solutions)
7. [Database-Related Fixes](#database-related-fixes)
8. [Diagnostic Decision Tree](#diagnostic-decision-tree)
9. [Emergency Rollback Procedures](#emergency-rollback-procedures)
10. [Preventive Measures](#preventive-measures)
11. [Security Considerations](#security-considerations)
12. [Performance Optimization](#performance-optimization)
13. [Real-World Examples](#real-world-examples)

---

## Quick Reference Emergency Checklist

### 🔴 CRITICAL (Immediate Action - 0-5 minutes)

- [ ] **Check if service is completely down or intermittent**
- [ ] **Review last deployment/change (within last 24 hours)**
- [ ] **Check server resource usage (CPU, Memory, Disk)**
  ```bash
  # Linux
  top
  df -h
  free -m

  # Windows
  taskmgr
  ```
- [ ] **Verify database connectivity**
  ```bash
  # MySQL
  mysqladmin -u root -p ping

  # PostgreSQL
  pg_isready -h localhost
  ```
- [ ] **Check error logs (last 100 lines)**
  ```bash
  # Apache
  tail -100 /var/log/apache2/error.log

  # Nginx
  tail -100 /var/log/nginx/error.log

  # Application logs
  tail -100 /var/log/application/error.log
  ```

### 🟡 HIGH PRIORITY (5-15 minutes)

- [ ] **Enable detailed error reporting (non-production only)**
- [ ] **Test with minimal configuration**
- [ ] **Verify file permissions**
  ```bash
  # Check web directory permissions
  ls -la /var/www/html/

  # Fix common permission issues
  chown -R www-data:www-data /var/www/html/
  chmod -R 755 /var/www/html/
  ```
- [ ] **Check disk space availability**
- [ ] **Review recent code commits**
- [ ] **Test database queries manually**

### 🟢 MEDIUM PRIORITY (15-30 minutes)

- [ ] **Review server configuration files**
- [ ] **Check for dependency updates/conflicts**
- [ ] **Verify SSL/TLS certificates**
- [ ] **Test API endpoints individually**
- [ ] **Review application logs for patterns**
- [ ] **Check third-party service status**

### ⚪ LOW PRIORITY (30+ minutes)

- [ ] **Perform comprehensive code review**
- [ ] **Run full system diagnostics**
- [ ] **Review and update documentation**
- [ ] **Implement additional monitoring**
- [ ] **Schedule post-mortem meeting**

---

## Understanding HTTP 500 Errors

### What is an HTTP 500 Error?

An HTTP 500 Internal Server Error indicates that the server encountered an unexpected condition that prevented it from fulfilling the request. Unlike 4xx errors (client-side), 5xx errors are **server-side issues**.

### Common HTTP 500 Variants

| Error Code | Name | Description |
|------------|------|-------------|
| **500** | Internal Server Error | Generic server error |
| **501** | Not Implemented | Server doesn't support functionality |
| **502** | Bad Gateway | Invalid response from upstream server |
| **503** | Service Unavailable | Server temporarily unavailable |
| **504** | Gateway Timeout | Upstream server timeout |
| **505** | HTTP Version Not Supported | Server doesn't support HTTP protocol version |

---

## Root Cause Analysis

### 1. Server-Side Script Errors

**Description:** Syntax errors, runtime exceptions, or logical errors in application code.

**Common Indicators:**
- Stack traces in error logs
- Specific line numbers referenced
- Consistent errors on specific endpoints
- Errors after recent code deployment

**Example Error Messages:**
```
PHP Fatal error: Uncaught Error: Call to undefined function
Python: NameError: name 'variable' is not defined
Node.js: ReferenceError: x is not defined
```

**Investigation Steps:**
```bash
# Check application error logs
tail -f /var/log/application/error.log | grep -i "fatal\|error\|exception"

# Enable debug mode (temporarily)
# PHP: display_errors = On in php.ini
# Python: DEBUG = True in settings.py
# Node.js: NODE_ENV=development
```

---

### 2. Database Connection Failures

**Description:** Unable to establish or maintain database connections.

**Common Causes:**
- Wrong credentials
- Database server down
- Connection pool exhausted
- Network connectivity issues
- Firewall blocking connections
- Max connection limit reached

**Symptoms:**
```
SQLSTATE[HY000] [2002] Connection refused
pymongo.errors.ServerSelectionTimeoutError
SequelizeConnectionError: connect ECONNREFUSED
```

**Quick Tests:**
```bash
# MySQL Connection Test
mysql -h hostname -u username -p -e "SELECT 1;"

# PostgreSQL Connection Test
psql -h hostname -U username -d database -c "SELECT 1;"

# MongoDB Connection Test
mongosh "mongodb://hostname:27017" --eval "db.adminCommand('ping')"

# Check active connections
# MySQL
mysqladmin -u root -p processlist

# PostgreSQL
psql -c "SELECT count(*) FROM pg_stat_activity;"
```

---

### 3. File Permission Problems

**Description:** Web server lacks necessary permissions to read/write files.

**Common Issues:**
- Incorrect file ownership
- Restrictive permissions (e.g., 600 instead of 644)
- SELinux/AppArmor blocking access
- Read-only file systems

**Diagnosis:**
```bash
# Check current permissions
ls -la /var/www/html/

# Common permission patterns
# Files: 644 (rw-r--r--)
# Directories: 755 (rwxr-xr-x)
# Writable directories: 775 (rwxrwxr-x)

# Check file ownership
stat /var/www/html/index.php

# Identify SELinux issues (RHEL/CentOS)
getenforce
ausearch -m avc -ts recent

# Fix permissions (be cautious)
find /var/www/html -type f -exec chmod 644 {} \;
find /var/www/html -type d -exec chmod 755 {} \;
chown -R www-data:www-data /var/www/html/
```

**Critical Files to Check:**
- Application configuration files
- Upload directories
- Cache directories
- Log files
- Session storage
- Temporary files

---

### 4. Resource Exhaustion

**Description:** Server running out of critical resources.

#### Memory Exhaustion

**Symptoms:**
```
PHP Fatal error: Allowed memory size of X bytes exhausted
Java: java.lang.OutOfMemoryError
Python: MemoryError
```

**Diagnosis:**
```bash
# Check memory usage
free -m
vmstat 1 10

# Monitor memory-intensive processes
ps aux --sort=-%mem | head -10

# Check PHP memory limit
php -i | grep memory_limit

# Monitor in real-time
watch -n 1 'free -m'
```

**Solutions:**
```bash
# Increase PHP memory limit
# In php.ini or .htaccess
memory_limit = 256M

# Java heap size
java -Xmx2g -Xms512m

# Node.js max old space
node --max-old-space-size=4096 app.js
```

#### CPU Exhaustion

**Diagnosis:**
```bash
# Check CPU usage
top -bn1 | head -20

# CPU-intensive processes
ps aux --sort=-%cpu | head -10

# Load average
uptime
cat /proc/loadavg
```

#### Disk Space Exhaustion

**Critical Check:**
```bash
# Check disk space
df -h

# Find large files
find /var -type f -size +100M -exec ls -lh {} \;

# Check inode usage
df -i

# Disk usage by directory
du -sh /* | sort -hr | head -10

# Clear common space hogs
# Log files
find /var/log -type f -name "*.log" -mtime +30 -delete

# Temporary files
rm -rf /tmp/*

# Package cache
apt-get clean  # Debian/Ubuntu
yum clean all  # RHEL/CentOS
```

#### Connection Pool Exhaustion

**Symptoms:**
```
Too many connections
Connection pool timeout
SQLSTATE[HY000]: General error: 2006 MySQL server has gone away
```

**Solutions:**
```sql
-- MySQL: Check max connections
SHOW VARIABLES LIKE 'max_connections';
SHOW STATUS LIKE 'Threads_connected';

-- Increase max connections
SET GLOBAL max_connections = 500;

-- PostgreSQL
SELECT count(*) FROM pg_stat_activity;
ALTER SYSTEM SET max_connections = 200;
```

---

### 5. Misconfigured Server Settings

**Description:** Incorrect web server or application configuration.

**Common Configuration Errors:**

#### Apache (.htaccess issues)
```apache
# Invalid syntax
RewrireEngine On  # Typo: should be "RewriteEngine"

# Missing required modules
# Enable mod_rewrite
a2enmod rewrite
systemctl restart apache2

# Check syntax
apachectl configtest
```

#### Nginx (nginx.conf issues)
```nginx
# Test configuration
nginx -t

# Common mistakes
# Missing semicolon
location / {
    try_files $uri $uri/ /index.php  # Missing semicolon
}

# Incorrect fastcgi_pass
fastcgi_pass unix:/var/run/php-fpm.sock;  # Verify socket exists
```

#### PHP Configuration (php.ini)
```ini
# Check current configuration
php -i | grep -i "configuration file"

# Common settings to verify
display_errors = Off  # Production
error_reporting = E_ALL
max_execution_time = 30
upload_max_filesize = 20M
post_max_size = 25M

# Restart PHP-FPM after changes
systemctl restart php8.1-fpm
```

---

## Systematic Debugging Process

### Step 1: Gather Initial Information

```bash
# Create debugging directory
mkdir -p /tmp/debug-$(date +%Y%m%d-%H%M%S)
cd /tmp/debug-*

# Collect system information
uname -a > system_info.txt
df -h > disk_space.txt
free -m > memory_info.txt
ps aux > process_list.txt

# Collect error logs
tail -1000 /var/log/apache2/error.log > apache_errors.txt
tail -1000 /var/log/nginx/error.log > nginx_errors.txt
tail -1000 /var/log/syslog > system_log.txt
```

### Step 2: Enable Detailed Error Reporting

**⚠️ WARNING: Only enable in development/staging environments!**

#### PHP
```php
<?php
// At the top of your PHP script or in php.ini
error_reporting(E_ALL);
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
ini_set('log_errors', 1);
ini_set('error_log', '/tmp/php_errors.log');
?>
```

#### Python (Django)
```python
# settings.py
DEBUG = True
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': '/tmp/django_debug.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
```

#### Python (Flask)
```python
# app.py
app.debug = True
app.config['PROPAGATE_EXCEPTIONS'] = True

import logging
logging.basicConfig(filename='/tmp/flask_errors.log', level=logging.DEBUG)
```

#### Node.js (Express)
```javascript
// app.js
process.env.NODE_ENV = 'development';

app.use((err, req, res, next) => {
  console.error('Error Details:', err);
  console.error('Stack Trace:', err.stack);
  res.status(500).json({
    error: err.message,
    stack: err.stack
  });
});
```

#### Java (Spring Boot)
```properties
# application.properties
logging.level.root=DEBUG
logging.file.name=/tmp/spring-boot-errors.log
server.error.include-message=always
server.error.include-stacktrace=always
```

#### .NET (ASP.NET Core)
```csharp
// Program.cs or Startup.cs
if (env.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}

// appsettings.Development.json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "Microsoft": "Debug"
    }
  }
}
```

### Step 3: Check Recent Code Changes

```bash
# Git: View recent commits
git log --oneline --since="24 hours ago"
git diff HEAD~1 HEAD

# Find recently modified files
find /var/www/html -type f -mtime -1 -ls

# SVN: Check recent changes
svn log -r HEAD:1 -l 10
svn diff -r PREV:HEAD
```

### Step 4: Verify Database Connectivity

```bash
# Create database connection test script

# PHP (test_db.php)
cat > /tmp/test_db.php << 'EOF'
<?php
$host = 'localhost';
$db = 'database_name';
$user = 'username';
$pass = 'password';

try {
    $pdo = new PDO("mysql:host=$host;dbname=$db", $user, $pass);
    echo "Database connection successful!\n";
} catch(PDOException $e) {
    echo "Connection failed: " . $e->getMessage() . "\n";
}
?>
EOF
php /tmp/test_db.php

# Python (test_db.py)
cat > /tmp/test_db.py << 'EOF'
import pymysql
try:
    conn = pymysql.connect(
        host='localhost',
        user='username',
        password='password',
        database='database_name'
    )
    print("Database connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
EOF
python3 /tmp/test_db.py

# Node.js (test_db.js)
cat > /tmp/test_db.js << 'EOF'
const mysql = require('mysql2');
const connection = mysql.createConnection({
  host: 'localhost',
  user: 'username',
  password: 'password',
  database: 'database_name'
});

connection.connect((err) => {
  if (err) {
    console.error('Connection failed:', err);
    return;
  }
  console.log('Database connection successful!');
  connection.end();
});
EOF
node /tmp/test_db.js
```

### Step 5: Review File and Directory Permissions

```bash
# Comprehensive permission audit script
cat > /tmp/check_permissions.sh << 'EOF'
#!/bin/bash

WEB_ROOT="/var/www/html"
WEB_USER="www-data"

echo "=== Permission Audit ==="
echo "Web Root: $WEB_ROOT"
echo "Web User: $WEB_USER"
echo ""

# Check ownership
echo "Files not owned by $WEB_USER:"
find $WEB_ROOT ! -user $WEB_USER -ls

# Check problematic permissions
echo ""
echo "World-writable files (security risk):"
find $WEB_ROOT -type f -perm -002 -ls

echo ""
echo "Files without read permission:"
find $WEB_ROOT -type f ! -perm -444 -ls

echo ""
echo "Directories without execute permission:"
find $WEB_ROOT -type d ! -perm -111 -ls

# Check critical directories
for dir in cache logs uploads sessions temp; do
    if [ -d "$WEB_ROOT/$dir" ]; then
        echo ""
        echo "Permissions for $dir:"
        ls -ld "$WEB_ROOT/$dir"
    fi
done
EOF

chmod +x /tmp/check_permissions.sh
/tmp/check_permissions.sh
```

### Step 6: Monitor Server Resources

```bash
# Real-time resource monitoring script
cat > /tmp/monitor_resources.sh << 'EOF'
#!/bin/bash

echo "=== Real-time Resource Monitor ==="
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    date
    echo ""
    echo "=== CPU Usage ==="
    top -bn1 | head -5

    echo ""
    echo "=== Memory Usage ==="
    free -h

    echo ""
    echo "=== Disk Usage ==="
    df -h | grep -v tmpfs

    echo ""
    echo "=== Top Processes (CPU) ==="
    ps aux --sort=-%cpu | head -6

    echo ""
    echo "=== Top Processes (Memory) ==="
    ps aux --sort=-%mem | head -6

    echo ""
    echo "=== Network Connections ==="
    netstat -an | grep ESTABLISHED | wc -l
    echo "Established connections"

    sleep 5
done
EOF

chmod +x /tmp/monitor_resources.sh
/tmp/monitor_resources.sh
```

### Step 7: Test Configuration Files

```bash
# Apache configuration test
apachectl configtest
apache2ctl -S  # Show virtual hosts

# Nginx configuration test
nginx -t
nginx -T  # Show full configuration

# PHP configuration test
php -i | grep -i "configuration file"
php -m  # Show loaded modules
php -v  # Show PHP version

# Test specific PHP syntax
php -l /path/to/script.php

# MySQL configuration test
mysqld --help --verbose | grep -A 1 'Default options'

# PostgreSQL configuration test
postgres --version
pg_config
```

---

## Language-Specific Debugging

### PHP Debugging

#### Common PHP Errors Leading to HTTP 500

**1. Fatal Errors**
```php
// Missing function
<?php
callUndefinedFunction();  // Fatal error
?>

// Memory exhaustion
<?php
$array = [];
while(true) {
    $array[] = str_repeat('x', 1000000);  // Will cause memory exhaustion
}
?>
```

**2. Parse Errors**
```php
<?php
// Missing semicolon
echo "Hello World"  // Parse error

// Mismatched brackets
function test() {
    if (true) {
        echo "test";
    // Missing closing brace
?>
```

#### PHP Debugging Commands

```bash
# Check PHP version and configuration
php -v
php -i | grep -i "error"
php -m | grep -i "pdo\|mysqli"

# Test PHP script syntax
php -l script.php

# Run PHP script from command line
php script.php

# Check PHP-FPM status
systemctl status php8.1-fpm
ps aux | grep php-fpm

# PHP-FPM error log
tail -f /var/log/php8.1-fpm.log

# Increase PHP-FPM logging
# In /etc/php/8.1/fpm/php-fpm.conf
log_level = debug
```

#### PHP Configuration Adjustments

```ini
# /etc/php/8.1/apache2/php.ini or /etc/php/8.1/fpm/php.ini

# Development settings
display_errors = On
display_startup_errors = On
error_reporting = E_ALL
log_errors = On
error_log = /var/log/php_errors.log

# Production settings
display_errors = Off
display_startup_errors = Off
error_reporting = E_ALL & ~E_DEPRECATED & ~E_STRICT
log_errors = On
error_log = /var/log/php_errors.log

# Resource limits
memory_limit = 256M
max_execution_time = 60
max_input_time = 60
upload_max_filesize = 20M
post_max_size = 25M

# Session configuration
session.save_path = /var/lib/php/sessions
session.gc_maxlifetime = 1440

# Restart required
systemctl restart apache2
# or
systemctl restart php8.1-fpm
```

#### PHP Error Handler

```php
<?php
// Custom error handler for debugging
set_error_handler(function($errno, $errstr, $errfile, $errline) {
    $error_message = "Error [$errno]: $errstr in $errfile on line $errline\n";
    error_log($error_message, 3, "/tmp/custom_php_errors.log");

    // In development, display error
    if (getenv('APP_ENV') === 'development') {
        echo "<pre>$error_message</pre>";
    }

    return false;  // Continue normal error handling
});

// Exception handler
set_exception_handler(function($exception) {
    $error_message = "Uncaught exception: " . $exception->getMessage() . "\n";
    $error_message .= "Stack trace:\n" . $exception->getTraceAsString();
    error_log($error_message, 3, "/tmp/custom_php_errors.log");

    if (getenv('APP_ENV') === 'development') {
        echo "<pre>$error_message</pre>";
    } else {
        http_response_code(500);
        echo "An error occurred. Please try again later.";
    }
});

// Shutdown handler for fatal errors
register_shutdown_function(function() {
    $error = error_get_last();
    if ($error !== null && in_array($error['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR])) {
        $error_message = "Fatal error: {$error['message']} in {$error['file']} on line {$error['line']}\n";
        error_log($error_message, 3, "/tmp/custom_php_errors.log");
    }
});
?>
```

---

### Python Debugging

#### Common Python Errors

**1. Import Errors**
```python
# ModuleNotFoundError
from non_existent_module import something

# Solution: Check installed packages
pip list
pip install required-package
```

**2. Database Connection Errors**
```python
# psycopg2.OperationalError
import psycopg2
conn = psycopg2.connect("dbname=test user=postgres password=wrong")
```

#### Python Debugging Commands

```bash
# Check Python version
python3 --version
which python3

# Check installed packages
pip list
pip freeze > requirements.txt

# Verify package installation
pip show package-name

# Django-specific
python manage.py check
python manage.py check --deploy

# Flask-specific
export FLASK_ENV=development
export FLASK_DEBUG=1
flask run

# Run with verbose logging
python -v script.py

# Check for syntax errors
python -m py_compile script.py

# Profile memory usage
python -m memory_profiler script.py
```

#### Python Configuration (Django)

```python
# settings.py - Development
DEBUG = True
ALLOWED_HOSTS = ['*']

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': '/tmp/django_debug.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Database debugging
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'dbname',
        'USER': 'user',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
        'CONN_MAX_AGE': 60,  # Connection pooling
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}
```

#### Python Configuration (Flask)

```python
# app.py
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Development configuration
if app.config['ENV'] == 'development':
    app.debug = True
    app.config['PROPAGATE_EXCEPTIONS'] = True

# Logging configuration
if not app.debug:
    file_handler = RotatingFileHandler('/tmp/flask_errors.log',
                                       maxBytes=10240000,
                                       backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return "Internal Server Error", 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception('Unhandled Exception: %s', e)
    return "An error occurred", 500
```

---

### Node.js Debugging

#### Common Node.js Errors

```javascript
// ReferenceError
console.log(undefinedVariable);

// TypeError
const obj = null;
obj.property;  // Cannot read property of null

// Database connection error
const mongoose = require('mongoose');
mongoose.connect('mongodb://wrong:27017/db', {
  useNewUrlParser: true,
  serverSelectionTimeoutMS: 5000
}).catch(err => console.error(err));
```

#### Node.js Debugging Commands

```bash
# Check Node.js version
node --version
npm --version

# Check installed packages
npm list --depth=0

# Audit for vulnerabilities
npm audit
npm audit fix

# Run with debugging
node --inspect app.js
node --inspect-brk app.js  # Break at start

# Run with increased memory
node --max-old-space-size=4096 app.js

# Enable verbose logging
NODE_ENV=development node app.js

# PM2 debugging
pm2 logs app-name --lines 100
pm2 describe app-name
pm2 restart app-name
```

#### Node.js Configuration (Express)

```javascript
// app.js
const express = require('express');
const app = express();

// Development vs Production
const isDev = process.env.NODE_ENV !== 'production';

// Logging middleware
const morgan = require('morgan');
if (isDev) {
  app.use(morgan('dev'));
} else {
  const fs = require('fs');
  const path = require('path');
  const accessLogStream = fs.createWriteStream(
    path.join(__dirname, 'access.log'),
    { flags: 'a' }
  );
  app.use(morgan('combined', { stream: accessLogStream }));
}

// Error logging
const winston = require('winston');
const logger = winston.createLogger({
  level: isDev ? 'debug' : 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

if (isDev) {
  logger.add(new winston.transports.Console({
    format: winston.format.simple()
  }));
}

// Global error handler
app.use((err, req, res, next) => {
  logger.error({
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method
  });

  if (isDev) {
    res.status(500).json({
      error: err.message,
      stack: err.stack
    });
  } else {
    res.status(500).json({
      error: 'Internal Server Error'
    });
  }
});

// Uncaught exception handler
process.on('uncaughtException', (err) => {
  logger.error('Uncaught Exception:', err);
  process.exit(1);
});

// Unhandled rejection handler
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

module.exports = app;
```

---

### Java Debugging

#### Common Java Errors

```java
// NullPointerException
String str = null;
str.length();  // NPE

// ClassNotFoundException
Class.forName("com.nonexistent.Class");

// OutOfMemoryError
List<byte[]> list = new ArrayList<>();
while(true) {
    list.add(new byte[1024 * 1024]);  // Will cause OOM
}
```

#### Java Debugging Commands

```bash
# Check Java version
java -version
javac -version

# Run with debugging enabled
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 -jar app.jar

# Increase heap size
java -Xmx2g -Xms512m -jar app.jar

# Garbage collection logging
java -Xlog:gc*:file=gc.log -jar app.jar

# Dump heap on OutOfMemoryError
java -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/heap_dump.hprof -jar app.jar

# Check running Java processes
jps -l

# Thread dump
jstack <pid> > thread_dump.txt

# Heap dump
jmap -dump:format=b,file=/tmp/heap_dump.hprof <pid>

# Memory usage
jstat -gc <pid> 1000 10
```

#### Java Configuration (Spring Boot)

```properties
# application.properties

# Logging
logging.level.root=INFO
logging.level.com.yourpackage=DEBUG
logging.file.name=/var/log/application.log
logging.pattern.file=%d{yyyy-MM-dd HH:mm:ss} - %msg%n

# Development mode
spring.devtools.restart.enabled=true
server.error.include-message=always
server.error.include-stacktrace=on_param

# Database
spring.datasource.url=jdbc:mysql://localhost:3306/dbname
spring.datasource.username=user
spring.datasource.password=password
spring.datasource.hikari.maximum-pool-size=10
spring.datasource.hikari.connection-timeout=30000

# JPA/Hibernate
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
```

---

### .NET Debugging

#### Common .NET Errors

```csharp
// NullReferenceException
string str = null;
int length = str.Length;  // NullReferenceException

// SqlException
using (SqlConnection conn = new SqlConnection("Server=invalid;"))
{
    conn.Open();  // SqlException
}
```

#### .NET Debugging Commands

```bash
# Check .NET version
dotnet --version
dotnet --list-sdks

# Run in development mode
export ASPNETCORE_ENVIRONMENT=Development
dotnet run

# Build with verbose output
dotnet build --verbosity detailed

# Check for errors
dotnet build

# Publish for production
dotnet publish -c Release
```

#### .NET Configuration (ASP.NET Core)

```csharp
// Program.cs
public class Program
{
    public static void Main(string[] args)
    {
        CreateHostBuilder(args).Build().Run();
    }

    public static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureLogging(logging =>
            {
                logging.ClearProviders();
                logging.AddConsole();
                logging.AddDebug();
                logging.AddEventSourceLogger();
            })
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
            });
}

// Startup.cs
public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
{
    if (env.IsDevelopment())
    {
        app.UseDeveloperExceptionPage();
    }
    else
    {
        app.UseExceptionHandler("/Error");
        app.UseHsts();
    }

    // Custom error handling
    app.Use(async (context, next) =>
    {
        try
        {
            await next();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unhandled exception");
            context.Response.StatusCode = 500;
            await context.Response.WriteAsync("An error occurred");
        }
    });
}
```

```json
// appsettings.Development.json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "Microsoft": "Information",
      "Microsoft.Hosting.Lifetime": "Information"
    }
  },
  "AllowedHosts": "*"
}
```

---

## Web Server-Specific Solutions

### Apache HTTP Server

#### Common Apache Issues

**1. .htaccess Errors**
```apache
# Syntax error - missing RewriteCond
RewriteEngine On
RewriteRule ^(.*)$ index.php [L]  # Missing conditions

# Correct version
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ index.php [QSA,L]
```

**2. Module Not Loaded**
```bash
# Check loaded modules
apache2ctl -M
# or
apachectl -M

# Enable required modules
a2enmod rewrite
a2enmod headers
a2enmod ssl
a2enmod proxy
a2enmod proxy_http

# Restart Apache
systemctl restart apache2
```

**3. Virtual Host Configuration**
```apache
# /etc/apache2/sites-available/example.conf
<VirtualHost *:80>
    ServerName example.com
    ServerAlias www.example.com
    DocumentRoot /var/www/html

    <Directory /var/www/html>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/example_error.log
    CustomLog ${APACHE_LOG_DIR}/example_access.log combined
</VirtualHost>

# Enable site
a2ensite example.conf

# Test configuration
apachectl configtest

# Reload
systemctl reload apache2
```

#### Apache Debugging Commands

```bash
# Test configuration syntax
apachectl configtest
apache2ctl -t

# Show loaded virtual hosts
apache2ctl -S

# Show compiled-in modules
apache2ctl -l

# View error log in real-time
tail -f /var/log/apache2/error.log

# Check Apache status
systemctl status apache2

# Increase log verbosity (temporarily)
# Add to httpd.conf or apache2.conf
LogLevel debug
```

#### Apache Performance Tuning

```apache
# /etc/apache2/mods-available/mpm_prefork.conf
<IfModule mpm_prefork_module>
    StartServers             5
    MinSpareServers          5
    MaxSpareServers         10
    MaxRequestWorkers      150
    MaxConnectionsPerChild   0
</IfModule>

# /etc/apache2/mods-available/mpm_worker.conf
<IfModule mpm_worker_module>
    StartServers             2
    MinSpareThreads         25
    MaxSpareThreads         75
    ThreadLimit             64
    ThreadsPerChild         25
    MaxRequestWorkers      150
    MaxConnectionsPerChild   0
</IfModule>
```

---

### Nginx

#### Common Nginx Issues

**1. Configuration Syntax Errors**
```nginx
# Missing semicolon
location / {
    try_files $uri $uri/ /index.php  # ERROR: Missing semicolon
}

# Correct version
location / {
    try_files $uri $uri/ /index.php?$query_string;
}
```

**2. PHP-FPM Connection Issues**
```nginx
# Check PHP-FPM socket
ls -l /var/run/php/php8.1-fpm.sock

# Nginx configuration
location ~ \.php$ {
    fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
    fastcgi_index index.php;
    include fastcgi_params;
    fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
}

# Alternative: TCP connection
location ~ \.php$ {
    fastcgi_pass 127.0.0.1:9000;
    # ... rest of configuration
}
```

**3. Server Block Configuration**
```nginx
# /etc/nginx/sites-available/example.com
server {
    listen 80;
    listen [::]:80;
    server_name example.com www.example.com;
    root /var/www/html;
    index index.php index.html;

    access_log /var/log/nginx/example_access.log;
    error_log /var/log/nginx/example_error.log;

    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }

    location ~ \.php$ {
        fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }

    location ~ /\.ht {
        deny all;
    }
}

# Enable site
ln -s /etc/nginx/sites-available/example.com /etc/nginx/sites-enabled/

# Test and reload
nginx -t
systemctl reload nginx
```

#### Nginx Debugging Commands

```bash
# Test configuration
nginx -t

# Show full configuration
nginx -T

# Check version and modules
nginx -V

# View error log
tail -f /var/log/nginx/error.log

# Check Nginx status
systemctl status nginx

# Reload configuration
systemctl reload nginx

# Restart Nginx
systemctl restart nginx

# Check PHP-FPM status
systemctl status php8.1-fpm

# Test PHP-FPM
cgi-fcgi -bind -connect /var/run/php/php8.1-fpm.sock
```

#### Nginx Performance Tuning

```nginx
# /etc/nginx/nginx.conf
user www-data;
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # Basic Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Buffer Settings
    client_body_buffer_size 128k;
    client_max_body_size 20m;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 8k;

    # Timeouts
    client_body_timeout 12;
    client_header_timeout 12;
    send_timeout 10;

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
}
```

---

### IIS (Internet Information Services)

#### Common IIS Issues

**1. Web.config Errors**
```xml
<!-- web.config -->
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <httpErrors errorMode="Detailed" />
        <asp scriptErrorSentToBrowser="true"/>

        <!-- URL Rewrite -->
        <rewrite>
            <rules>
                <rule name="Main Rule" stopProcessing="true">
                    <match url=".*" />
                    <conditions logicalGrouping="MatchAll">
                        <add input="{REQUEST_FILENAME}" matchType="IsFile" negate="true" />
                        <add input="{REQUEST_FILENAME}" matchType="IsDirectory" negate="true" />
                    </conditions>
                    <action type="Rewrite" url="index.php" />
                </rule>
            </rules>
        </rewrite>
    </system.webServer>
</configuration>
```

**2. Application Pool Issues**
```powershell
# Import IIS module
Import-Module WebAdministration

# Check application pool status
Get-IISAppPool

# Restart application pool
Restart-WebAppPool -Name "DefaultAppPool"

# Set application pool identity
Set-ItemProperty IIS:\AppPools\DefaultAppPool -Name processModel.identityType -Value 3

# Increase queue length
Set-ItemProperty IIS:\AppPools\DefaultAppPool -Name queueLength -Value 2000

# Set recycling
Set-ItemProperty IIS:\AppPools\DefaultAppPool -Name recycling.periodicRestart.time -Value "00:00:00"
```

#### IIS Debugging Commands

```powershell
# Check IIS status
Get-Service W3SVC

# Start IIS
Start-Service W3SVC

# View application pools
Get-IISAppPool | Select-Object Name, Status

# View sites
Get-IISSite

# Enable detailed errors (temporary)
# In web.config or IIS Manager

# Check failed request tracing
# Enable in IIS Manager > Failed Request Tracing Rules

# View event logs
Get-EventLog -LogName Application -Source "IIS*" -Newest 50
```

---

### LiteSpeed

#### Common LiteSpeed Issues

**1. .htaccess Compatibility**
```apache
# LiteSpeed supports most Apache directives
RewriteEngine On
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ index.php [QSA,L]

# LiteSpeed-specific cache control
<IfModule LiteSpeed>
    CacheLookup on
</IfModule>
```

**2. PHP Configuration**
```bash
# Check PHP version
/usr/local/lsws/fcgi-bin/lsphp -v

# Edit PHP configuration
vi /usr/local/lsws/lsphp81/etc/php/8.1/litespeed/php.ini

# Restart LiteSpeed
/usr/local/lsws/bin/lswsctrl restart
```

#### LiteSpeed Debugging Commands

```bash
# Check LiteSpeed status
/usr/local/lsws/bin/lswsctrl status

# Start LiteSpeed
/usr/local/lsws/bin/lswsctrl start

# Restart LiteSpeed
/usr/local/lsws/bin/lswsctrl restart

# Graceful restart
/usr/local/lsws/bin/lswsctrl graceful

# View error log
tail -f /usr/local/lsws/logs/error.log

# View access log
tail -f /usr/local/lsws/logs/access.log
```

---

## Database-Related Fixes

### MySQL/MariaDB

#### Connection Issues

**1. Authentication Errors**
```bash
# Test connection
mysql -u username -p -h hostname

# Common errors and fixes
# ERROR 1045 (28000): Access denied
# Solution: Reset password or check privileges

# Reset root password
sudo mysqld_safe --skip-grant-tables &
mysql -u root
UPDATE mysql.user SET authentication_string=PASSWORD('new_password') WHERE User='root';
FLUSH PRIVILEGES;

# Grant privileges
GRANT ALL PRIVILEGES ON database.* TO 'user'@'localhost' IDENTIFIED BY 'password';
FLUSH PRIVILEGES;
```

**2. Connection Limit Issues**
```sql
-- Check current connections
SHOW PROCESSLIST;
SHOW STATUS LIKE 'Threads_connected';

-- Check max connections
SHOW VARIABLES LIKE 'max_connections';

-- Increase max connections (temporary)
SET GLOBAL max_connections = 500;

-- Increase max connections (permanent)
-- Add to my.cnf or my.ini
[mysqld]
max_connections = 500

-- Kill specific connection
KILL CONNECTION_ID;
```

**3. Database Lock Issues**
```sql
-- Check for locked tables
SHOW OPEN TABLES WHERE In_use > 0;

-- Check for long-running queries
SELECT * FROM information_schema.processlist WHERE time > 60;

-- Kill long-running query
KILL QUERY process_id;

-- Check InnoDB status
SHOW ENGINE INNODB STATUS;
```

#### MySQL Performance Tuning

```ini
# /etc/mysql/my.cnf or /etc/my.cnf
[mysqld]
# Connection Settings
max_connections = 500
connect_timeout = 10
wait_timeout = 600
max_connect_errors = 100

# Buffer Settings
innodb_buffer_pool_size = 1G
innodb_log_file_size = 256M
innodb_log_buffer_size = 8M
key_buffer_size = 256M

# Query Cache (MySQL 5.7 and earlier)
query_cache_type = 1
query_cache_size = 64M
query_cache_limit = 2M

# Logging
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 2

# Restart MySQL
systemctl restart mysql
```

---

### PostgreSQL

#### Connection Issues

**1. Authentication Configuration**
```bash
# Check pg_hba.conf
sudo vi /etc/postgresql/14/main/pg_hba.conf

# Allow local connections
local   all             all                                     peer
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5

# Reload configuration
sudo systemctl reload postgresql
```

**2. Connection Pooling**
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Check max connections
SHOW max_connections;

-- Increase max connections
ALTER SYSTEM SET max_connections = 200;
-- Restart required
```

**3. Lock Monitoring**
```sql
-- Check for locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Check blocking queries
SELECT pid, usename, pg_blocking_pids(pid) AS blocked_by, query
FROM pg_stat_activity
WHERE cardinality(pg_blocking_pids(pid)) > 0;

-- Terminate specific connection
SELECT pg_terminate_backend(pid);
```

#### PostgreSQL Performance Tuning

```bash
# /etc/postgresql/14/main/postgresql.conf

# Connection Settings
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
maintenance_work_mem = 128MB

# WAL Settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9

# Logging
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000

# Restart PostgreSQL
sudo systemctl restart postgresql
```

---

### MongoDB

#### Connection Issues

**1. Authentication Errors**
```bash
# Connect to MongoDB
mongosh "mongodb://localhost:27017"

# With authentication
mongosh "mongodb://username:password@localhost:27017/database"

# Create user
use admin
db.createUser({
  user: "admin",
  pwd: "password",
  roles: [ { role: "userAdminAnyDatabase", db: "admin" } ]
})
```

**2. Connection Pool Exhaustion**
```javascript
// Node.js with Mongoose
const mongoose = require('mongoose');

mongoose.connect('mongodb://localhost:27017/mydb', {
  maxPoolSize: 50,
  minPoolSize: 10,
  serverSelectionTimeoutMS: 5000,
  socketTimeoutMS: 45000,
  family: 4
});

// Monitor connections
mongoose.connection.on('connected', () => {
  console.log('MongoDB connected');
});

mongoose.connection.on('error', (err) => {
  console.error('MongoDB connection error:', err);
});
```

**3. Performance Issues**
```javascript
// Check slow queries
db.setProfilingLevel(1, { slowms: 100 });
db.system.profile.find().limit(10).sort({ ts: -1 }).pretty();

// Check current operations
db.currentOp();

// Kill operation
db.killOp(opid);

// Check database stats
db.stats();

// Create indexes for better performance
db.collection.createIndex({ field: 1 });
```

---

### SQL Server

#### Connection Issues

**1. Connection String Errors**
```csharp
// C# connection string examples
string connectionString = "Server=localhost;Database=myDB;User Id=sa;Password=password;";

// With integrated security
string connectionString = "Server=localhost;Database=myDB;Integrated Security=true;";

// With connection pooling
string connectionString = "Server=localhost;Database=myDB;User Id=sa;Password=password;Min Pool Size=5;Max Pool Size=100;";
```

**2. Check Connection Status**
```sql
-- Check active connections
SELECT
    DB_NAME(dbid) as DatabaseName,
    COUNT(dbid) as NumberOfConnections,
    loginame
FROM sys.sysprocesses
WHERE dbid > 0
GROUP BY dbid, loginame;

-- Check blocked processes
EXEC sp_who2;

-- Kill specific process
KILL 52; -- Replace with SPID
```

**3. Deadlock Detection**
```sql
-- Enable trace flags for deadlock information
DBCC TRACEON (1222, -1);

-- View deadlock information
-- Check SQL Server Error Log

-- Query to find deadlocks
SELECT * FROM sys.dm_exec_requests WHERE blocking_session_id <> 0;
```

---

## Diagnostic Decision Tree

### HTTP 500 Error Diagnostic Flowchart

```
START: HTTP 500 Error Detected
│
├─→ Is the entire site down?
│   ├─→ YES
│   │   ├─→ Check server resources (CPU, Memory, Disk)
│   │   │   ├─→ Resources exhausted? → Free up resources → Restart services
│   │   │   └─→ Resources OK → Check web server status
│   │   │       ├─→ Service stopped? → Start service → Check logs
│   │   │       └─→ Service running → Check configuration files
│   │   │           ├─→ Syntax error? → Fix error → Restart
│   │   │           └─→ No syntax error → Check recent changes → ROLLBACK
│   │   │
│   └─→ NO (Intermittent or specific pages)
│       │
│       ├─→ Error on specific endpoints only?
│       │   ├─→ YES → Check application code
│       │   │   ├─→ Recent deployment? → Review changes → Test rollback
│       │   │   ├─→ Check error logs → Identify exception → Fix code
│       │   │   └─→ Database query issue? → Optimize query → Add indexes
│       │   │
│       │   └─→ NO → Random/intermittent errors
│       │       ├─→ Check database connections
│       │       │   ├─→ Connection pool exhausted? → Increase pool size
│       │       │   ├─→ Database timeout? → Optimize queries → Increase timeout
│       │       │   └─→ Database server down? → Restart database
│       │       │
│       │       ├─→ Check memory leaks
│       │       │   └─→ Memory increasing over time? → Profile application → Fix leaks
│       │       │
│       │       └─→ Check third-party services
│       │           └─→ External API timeout? → Implement retries → Add circuit breaker
│       │
├─→ Check Error Logs
│   ├─→ Clear error message found?
│   │   ├─→ YES → Follow error-specific solution below
│   │   └─→ NO → Enable detailed error reporting (dev only)
│   │
├─→ Common Error Patterns:
│   │
│   ├─→ "Fatal error: Allowed memory size exhausted"
│   │   → Increase memory_limit in php.ini → Optimize code
│   │
│   ├─→ "Connection refused" / "Too many connections"
│   │   → Check database server → Increase max_connections → Fix connection leaks
│   │
│   ├─→ "Permission denied"
│   │   → Fix file permissions → Check SELinux → Verify ownership
│   │
│   ├─→ "Maximum execution time exceeded"
│   │   → Increase max_execution_time → Optimize slow code
│   │
│   ├─→ "Class not found" / "Module not found"
│   │   → Install missing dependencies → Check autoloader → Verify paths
│   │
│   ├─→ "Syntax error" / "Parse error"
│   │   → Review recent code changes → Fix syntax → Test thoroughly
│   │
│   └─→ "Cannot write to file/directory"
│       → Fix permissions → Check disk space → Verify paths
│
└─→ RESOLUTION STEPS:
    1. Implement fix based on diagnosis
    2. Test in staging environment
    3. Deploy to production
    4. Monitor for 24-48 hours
    5. Document incident and resolution
    6. Implement preventive measures
```

### Quick Diagnosis Decision Matrix

| Symptom | Likely Cause | First Action | Priority |
|---------|-------------|--------------|----------|
| **Entire site down** | Web server crashed, config error | Check service status, restart | 🔴 CRITICAL |
| **Specific endpoints fail** | Code error, missing dependency | Check application logs | 🔴 CRITICAL |
| **Random intermittent errors** | Resource exhaustion, connection pool | Monitor resources, check connections | 🟡 HIGH |
| **After deployment** | New code bug, config change | Review changes, rollback if needed | 🔴 CRITICAL |
| **Gradual degradation** | Memory leak, connection leak | Profile application, monitor trends | 🟡 HIGH |
| **Peak hours only** | Insufficient resources, connection limits | Scale resources, optimize queries | 🟡 HIGH |
| **External API calls fail** | Third-party service issue, timeout | Check service status, add fallback | 🟢 MEDIUM |
| **Database operations slow** | Missing indexes, query optimization | Analyze slow queries, add indexes | 🟡 HIGH |
| **File upload fails** | Permission, disk space, size limit | Check permissions, disk, config | 🟢 MEDIUM |
| **After server restart** | Configuration issue, startup script | Check startup logs, verify config | 🟡 HIGH |

---

## Emergency Rollback Procedures

### Git-Based Deployment

#### Quick Rollback
```bash
# Method 1: Revert to previous commit
cd /var/www/html
git log --oneline -5  # Find previous working commit
git checkout <previous-commit-hash>

# Restart services
systemctl restart apache2
# or
systemctl restart php8.1-fpm
systemctl restart nginx

# Method 2: Use git revert (creates new commit)
git revert <bad-commit-hash>
git push origin main

# Method 3: Reset to previous tag/release
git fetch --tags
git checkout tags/v1.2.3
```

#### Detailed Rollback Process
```bash
#!/bin/bash
# rollback.sh - Emergency rollback script

DEPLOY_DIR="/var/www/html"
BACKUP_DIR="/var/backups/webroot"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "Starting emergency rollback at $TIMESTAMP"

# Create backup of current state
cp -r $DEPLOY_DIR $BACKUP_DIR/before-rollback-$TIMESTAMP

# Get previous working commit
cd $DEPLOY_DIR
PREVIOUS_COMMIT=$(git log --oneline -2 | tail -1 | awk '{print $1}')

echo "Rolling back to commit: $PREVIOUS_COMMIT"
git checkout $PREVIOUS_COMMIT

# Clear application cache
rm -rf cache/* logs/cache/*

# Composer/npm dependencies (if needed)
if [ -f "composer.json" ]; then
    composer install --no-dev --optimize-autoloader
fi

if [ -f "package.json" ]; then
    npm install --production
fi

# Restart services
systemctl restart apache2
systemctl restart php8.1-fpm

# Verify rollback
sleep 5
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost)

if [ "$HTTP_CODE" = "200" ]; then
    echo "Rollback successful! Site responding with HTTP 200"
else
    echo "WARNING: Site responding with HTTP $HTTP_CODE"
fi

echo "Rollback completed at $(date +%Y%m%d-%H%M%S)"
```

---

### Database Rollback

#### MySQL Database Rollback
```bash
#!/bin/bash
# Database rollback script

DB_NAME="production_db"
DB_USER="dbuser"
DB_PASS="password"
BACKUP_FILE="/var/backups/mysql/latest_backup.sql"

echo "Starting database rollback..."

# Create safety backup
mysqldump -u $DB_USER -p$DB_PASS $DB_NAME > /var/backups/mysql/pre-rollback-$(date +%Y%m%d-%H%M%S).sql

# Restore from backup
mysql -u $DB_USER -p$DB_PASS $DB_NAME < $BACKUP_FILE

echo "Database rollback completed"
```

---

### Docker-Based Rollback

```bash
# Rollback to previous container version
docker ps  # Get current container ID

# Stop current container
docker stop <container-id>

# Get previous image
docker images | grep myapp

# Start previous version
docker run -d --name myapp -p 80:80 myapp:v1.2.3

# Or use docker-compose
cd /opt/myapp
git checkout tags/v1.2.3
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs -f --tail=100
```

---

### Configuration File Rollback

```bash
#!/bin/bash
# Config rollback script

CONFIG_DIR="/etc/nginx"
BACKUP_DIR="/etc/nginx/backups"

# Rollback Nginx config
cp $BACKUP_DIR/nginx.conf.backup $CONFIG_DIR/nginx.conf
cp $BACKUP_DIR/sites-available/example.com.backup $CONFIG_DIR/sites-available/example.com

# Test configuration
nginx -t

if [ $? -eq 0 ]; then
    systemctl reload nginx
    echo "Configuration rollback successful"
else
    echo "Configuration test failed! Manual intervention required"
    exit 1
fi
```

---

## Preventive Measures

### 1. Implement Proper Error Handling

#### PHP Error Handling
```php
<?php
// config/error_handler.php

// Production error handler
function productionErrorHandler($errno, $errstr, $errfile, $errline) {
    // Log error details
    error_log(sprintf(
        "[%s] Error [%d]: %s in %s on line %d",
        date('Y-m-d H:i:s'),
        $errno,
        $errstr,
        $errfile,
        $errline
    ), 3, "/var/log/application/errors.log");

    // Display generic message to user
    if ($errno === E_ERROR || $errno === E_USER_ERROR) {
        http_response_code(500);
        include 'error_pages/500.html';
        exit(1);
    }

    return false;
}

// Exception handler
function productionExceptionHandler($exception) {
    error_log(sprintf(
        "[%s] Uncaught Exception: %s\nStack trace:\n%s",
        date('Y-m-d H:i:s'),
        $exception->getMessage(),
        $exception->getTraceAsString()
    ), 3, "/var/log/application/exceptions.log");

    http_response_code(500);
    include 'error_pages/500.html';
    exit(1);
}

// Register handlers
set_error_handler('productionErrorHandler');
set_exception_handler('productionExceptionHandler');
```

#### Python (Flask) Error Handling
```python
from flask import Flask, render_template
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
handler = RotatingFileHandler('errors.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)

@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error(f'Server Error: {error}', exc_info=True)
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.error(f'Unexpected error: {error}', exc_info=True)
    return render_template('500.html'), 500
```

#### Node.js Error Handling
```javascript
const express = require('express');
const winston = require('winston');
const app = express();

// Configure Winston logger
const logger = winston.createLogger({
  level: 'error',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error({
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip
  });

  res.status(500).render('error', {
    message: 'Internal Server Error'
  });
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  logger.error('Uncaught Exception:', err);
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection:', { reason, promise });
});
```

---

### 2. Monitoring and Alerting Systems

#### Application Performance Monitoring (APM)

**Using New Relic**
```php
// PHP
<?php
// Add to your bootstrap file
if (extension_loaded('newrelic')) {
    newrelic_set_appname("MyApp");
}
```

**Using Datadog**
```javascript
// Node.js
const tracer = require('dd-trace').init({
  hostname: 'localhost',
  port: 8126,
  service: 'my-application',
  env: 'production'
});
```

#### Server Monitoring Script
```bash
#!/bin/bash
# monitor.sh - Server health monitoring

ALERT_EMAIL="admin@example.com"
CPU_THRESHOLD=80
MEM_THRESHOLD=80
DISK_THRESHOLD=85

# Check CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'.' -f1)
if [ $CPU_USAGE -gt $CPU_THRESHOLD ]; then
    echo "High CPU usage: $CPU_USAGE%" | mail -s "CPU Alert" $ALERT_EMAIL
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d'.' -f1)
if [ $MEM_USAGE -gt $MEM_THRESHOLD ]; then
    echo "High memory usage: $MEM_USAGE%" | mail -s "Memory Alert" $ALERT_EMAIL
fi

# Check disk usage
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
if [ $DISK_USAGE -gt $DISK_THRESHOLD ]; then
    echo "High disk usage: $DISK_USAGE%" | mail -s "Disk Alert" $ALERT_EMAIL
fi

# Check web server
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost)
if [ "$HTTP_CODE" != "200" ]; then
    echo "Web server returned HTTP $HTTP_CODE" | mail -s "Web Server Alert" $ALERT_EMAIL
fi

# Add to crontab to run every 5 minutes
# */5 * * * * /usr/local/bin/monitor.sh
```

---

### 3. Regular Backups

#### Automated Backup Script
```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/var/backups"
WEB_DIR="/var/www/html"
DB_NAME="production_db"
DB_USER="backup_user"
DB_PASS="backup_password"
DATE=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS=30

# Create backup directories
mkdir -p $BACKUP_DIR/web
mkdir -p $BACKUP_DIR/database

# Backup web files
tar -czf $BACKUP_DIR/web/webroot-$DATE.tar.gz -C $WEB_DIR .

# Backup database
mysqldump -u $DB_USER -p$DB_PASS $DB_NAME | gzip > $BACKUP_DIR/database/db-$DATE.sql.gz

# Remove old backups
find $BACKUP_DIR/web -name "webroot-*.tar.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR/database -name "db-*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Upload to remote storage (optional)
# aws s3 sync $BACKUP_DIR s3://my-backups/$(hostname)/

echo "Backup completed: $DATE"

# Add to crontab for daily backups at 2 AM
# 0 2 * * * /usr/local/bin/backup.sh >> /var/log/backup.log 2>&1
```

---

### 4. Version Control Best Practices

```bash
# .gitignore for web projects
# Prevent committing sensitive files

# Environment files
.env
.env.local
.env.production

# Configuration
config/database.php
config/production.php

# Dependencies
/vendor/
/node_modules/

# Cache and logs
/cache/
/logs/
*.log

# Uploads
/uploads/
/public/uploads/

# IDE files
.idea/
.vscode/
*.swp
*.swo
```

---

### 5. Staging Environment Setup

```bash
#!/bin/bash
# setup_staging.sh - Create staging environment

# Clone production environment
rsync -av --exclude='logs' --exclude='cache' /var/www/production/ /var/www/staging/

# Copy database
mysqldump -u root -p production_db | mysql -u root -p staging_db

# Update staging configuration
cd /var/www/staging
cp .env.production .env.staging

# Update environment variables
sed -i 's/APP_ENV=production/APP_ENV=staging/g' .env.staging
sed -i 's/DB_DATABASE=production_db/DB_DATABASE=staging_db/g' .env.staging

# Set appropriate permissions
chown -R www-data:www-data /var/www/staging
chmod -R 755 /var/www/staging

echo "Staging environment ready"
```

---

### 6. Documentation Standards

#### Server Configuration Documentation Template
```markdown
# Server Configuration Documentation

## Server Details
- **Hostname**: web-prod-01
- **IP Address**: 192.168.1.100
- **OS**: Ubuntu 22.04 LTS
- **Web Server**: Nginx 1.24.0
- **PHP Version**: 8.1.12
- **Database**: MySQL 8.0.32

## Directory Structure
```
/var/www/
├── html/              # Web root
├── logs/              # Application logs
├── cache/             # Cache files
└── uploads/           # User uploads
```

## Important File Locations
- **Nginx Config**: /etc/nginx/nginx.conf
- **PHP-FPM Config**: /etc/php/8.1/fpm/php.ini
- **MySQL Config**: /etc/mysql/my.cnf
- **Application Config**: /var/www/html/config/

## Backup Schedule
- **Database**: Daily at 2:00 AM
- **Web Files**: Daily at 3:00 AM
- **Retention**: 30 days

## Monitoring
- **APM**: New Relic
- **Server Monitoring**: Prometheus + Grafana
- **Log Aggregation**: ELK Stack

## Contact Information
- **Primary Admin**: admin@example.com
- **On-Call**: oncall@example.com
- **Emergency**: +1-555-0123

## Change Log
| Date | Change | Person | Rollback Point |
|------|--------|--------|----------------|
| 2025-12-01 | Updated PHP to 8.1 | John | Tag: v1.2.3 |
| 2025-11-15 | Added Redis cache | Jane | Commit: abc123 |
```

---

## Security Considerations

### Exposing Error Details - Security Implications

#### ⚠️ CRITICAL: Never Expose Detailed Errors in Production

**Bad Practice (NEVER in Production):**
```php
<?php
// DANGEROUS - Exposes system information
ini_set('display_errors', 1);
error_reporting(E_ALL);
?>
```

**Good Practice (Production):**
```php
<?php
// Safe production settings
ini_set('display_errors', 0);
error_reporting(E_ALL);
ini_set('log_errors', 1);
ini_set('error_log', '/var/log/php/errors.log');

// Custom error page
set_exception_handler(function($e) {
    error_log($e->getMessage() . "\n" . $e->getTraceAsString());
    http_response_code(500);
    include '/var/www/errors/500.html';
    exit;
});
?>
```

### Information Disclosure Risks

**What Attackers Can Learn from Detailed Errors:**

1. **Directory Structure**
   ```
   Fatal error in /var/www/html/includes/database.php on line 42
   → Reveals: Directory layout, file naming conventions
   ```

2. **Database Information**
   ```
   SQLSTATE[28000]: Access denied for user 'webapp'@'localhost' using password YES
   → Reveals: Database username, hostname, authentication method
   ```

3. **Software Versions**
   ```
   PHP Fatal error: Uncaught Error in PHP 8.1.12
   → Reveals: Exact PHP version (may have known vulnerabilities)
   ```

4. **Code Logic**
   ```
   Stack trace showing function calls and parameters
   → Reveals: Application architecture, business logic
   ```

### Secure Error Handling Implementation

#### Generic Error Pages

**Create Custom 500.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Temporarily Unavailable</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background-color: #f5f5f5;
        }
        .error-container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #d32f2f;
            margin-bottom: 20px;
        }
        p {
            color: #666;
            line-height: 1.6;
        }
        .error-code {
            color: #999;
            font-size: 14px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>We'll be back soon!</h1>
        <p>Sorry for the inconvenience. We're performing some maintenance at the moment.</p>
        <p>Please try again in a few minutes.</p>
        <div class="error-code">
            Error Reference: <?php echo uniqid('ERR-'); ?>
        </div>
    </div>
</body>
</html>
```

#### Secure Logging

```php
<?php
// Secure error logging with sanitization

function secureErrorLog($message, $context = []) {
    // Remove sensitive data
    $sanitized = preg_replace('/password["\']?\s*[:=]\s*["\']?[^"\'\s,}]+/', 'password=***', $message);
    $sanitized = preg_replace('/api[_-]?key["\']?\s*[:=]\s*["\']?[^"\'\s,}]+/', 'api_key=***', $sanitized);
    $sanitized = preg_replace('/token["\']?\s*[:=]\s*["\']?[^"\'\s,}]+/', 'token=***', $sanitized);

    // Log with context
    $logEntry = sprintf(
        "[%s] %s | Context: %s\n",
        date('Y-m-d H:i:s'),
        $sanitized,
        json_encode($context)
    );

    error_log($logEntry, 3, '/var/log/application/secure.log');
}
```

### Rate Limiting Error Responses

```nginx
# Nginx - Rate limit error page requests to prevent reconnaissance
limit_req_zone $binary_remote_addr zone=error_limit:10m rate=10r/m;

location = /500.html {
    limit_req zone=error_limit burst=5;
    internal;
}
```

### Security Headers for Error Pages

```nginx
# Add security headers to error responses
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer" always;
```

---

## Performance Optimization

### 1. Code-Level Optimizations

#### Database Query Optimization

**Bad Practice:**
```php
<?php
// N+1 Query Problem
$users = $db->query("SELECT * FROM users");
foreach ($users as $user) {
    // Additional query for each user
    $posts = $db->query("SELECT * FROM posts WHERE user_id = " . $user['id']);
}
?>
```

**Good Practice:**
```php
<?php
// Single optimized query with JOIN
$query = "
    SELECT u.*, p.*
    FROM users u
    LEFT JOIN posts p ON u.id = p.user_id
";
$results = $db->query($query);
?>
```

#### Implement Query Caching

```php
<?php
class QueryCache {
    private $redis;

    public function __construct() {
        $this->redis = new Redis();
        $this->redis->connect('127.0.0.1', 6379);
    }

    public function get($key, $callback, $ttl = 3600) {
        $cached = $this->redis->get($key);

        if ($cached !== false) {
            return json_decode($cached, true);
        }

        $data = $callback();
        $this->redis->setex($key, $ttl, json_encode($data));

        return $data;
    }
}

// Usage
$cache = new QueryCache();
$users = $cache->get('users:active', function() use ($db) {
    return $db->query("SELECT * FROM users WHERE active = 1");
}, 1800);
?>
```

#### Lazy Loading

```javascript
// Node.js - Lazy load modules
const express = require('express');
const app = express();

// Load heavy modules only when needed
app.get('/report', async (req, res) => {
    const reportGenerator = require('./heavy-report-module');
    const report = await reportGenerator.generate();
    res.json(report);
});
```

### 2. Server-Level Optimizations

#### Enable OPcache (PHP)

```ini
# /etc/php/8.1/fpm/conf.d/10-opcache.ini
opcache.enable=1
opcache.enable_cli=0
opcache.memory_consumption=256
opcache.interned_strings_buffer=16
opcache.max_accelerated_files=10000
opcache.max_wasted_percentage=10
opcache.validate_timestamps=0  # Production only
opcache.revalidate_freq=0
opcache.save_comments=0
opcache.fast_shutdown=1
```

#### Connection Pooling

**Database Connection Pool (Node.js)**
```javascript
const mysql = require('mysql2/promise');

const pool = mysql.createPool({
    host: 'localhost',
    user: 'dbuser',
    password: 'password',
    database: 'mydb',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
    enableKeepAlive: true,
    keepAliveInitialDelay: 0
});

// Usage
async function getUsers() {
    const connection = await pool.getConnection();
    try {
        const [rows] = await connection.query('SELECT * FROM users');
        return rows;
    } finally {
        connection.release();
    }
}
```

#### Implement Caching Strategy

**Multi-Layer Cache (PHP)**
```php
<?php
class CacheManager {
    private $memcached;
    private $redis;

    public function __construct() {
        $this->memcached = new Memcached();
        $this->memcached->addServer('localhost', 11211);

        $this->redis = new Redis();
        $this->redis->connect('127.0.0.1', 6379);
    }

    public function get($key) {
        // L1: Memcached (fastest)
        $value = $this->memcached->get($key);
        if ($value !== false) {
            return $value;
        }

        // L2: Redis (persistent)
        $value = $this->redis->get($key);
        if ($value !== false) {
            $this->memcached->set($key, $value, 3600);
            return $value;
        }

        return null;
    }

    public function set($key, $value, $ttl = 3600) {
        $this->memcached->set($key, $value, $ttl);
        $this->redis->setex($key, $ttl, $value);
    }
}
?>
```

### 3. Resource Management

#### Memory Management

```php
<?php
// Process large datasets in chunks
function processBigFile($filename) {
    $handle = fopen($filename, 'r');
    if (!$handle) {
        throw new Exception("Cannot open file");
    }

    while (($line = fgets($handle)) !== false) {
        // Process line by line instead of loading entire file
        processLine($line);

        // Clear unnecessary variables
        unset($line);
    }

    fclose($handle);

    // Force garbage collection
    gc_collect_cycles();
}
?>
```

#### Connection Timeout Settings

```php
<?php
// Set appropriate timeouts
$context = stream_context_create([
    'http' => [
        'timeout' => 10,  // 10 seconds
        'ignore_errors' => true
    ]
]);

try {
    $result = file_get_contents('https://api.example.com/data', false, $context);
} catch (Exception $e) {
    error_log("API timeout: " . $e->getMessage());
    // Fallback logic
}
?>
```

### 4. Load Balancing

#### Nginx Load Balancer Configuration

```nginx
# /etc/nginx/nginx.conf
upstream backend {
    least_conn;  # Use least connections algorithm

    server backend1.example.com:8080 max_fails=3 fail_timeout=30s;
    server backend2.example.com:8080 max_fails=3 fail_timeout=30s;
    server backend3.example.com:8080 max_fails=3 fail_timeout=30s;

    # Health check
    keepalive 32;
}

server {
    listen 80;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

---

## Real-World Examples

### Example 1: Out of Memory Error

**Error Message:**
```
PHP Fatal error: Allowed memory size of 134217728 bytes exhausted
(tried to allocate 20480 bytes) in /var/www/html/process.php on line 45
```

**Diagnosis:**
```bash
# Check current PHP memory limit
php -i | grep memory_limit
# Output: memory_limit => 128M => 128M

# Check script execution
php -d memory_limit=-1 process.php  # Test with unlimited memory
```

**Root Cause:**
Script loading entire dataset into memory instead of processing in chunks.

**Solution:**
```php
<?php
// Before (Bad): Load everything into memory
$data = file_get_contents('large_file.csv');
$lines = explode("\n", $data);
foreach ($lines as $line) {
    process($line);
}

// After (Good): Process line by line
$handle = fopen('large_file.csv', 'r');
while (($line = fgets($handle)) !== false) {
    process($line);
}
fclose($handle);
?>
```

**Prevention:**
- Set appropriate memory limits based on application needs
- Implement chunked processing for large datasets
- Use generators for memory-efficient iteration
- Monitor memory usage in production

---

### Example 2: Database Connection Exhaustion

**Error Message:**
```
SQLSTATE[HY000] [1040] Too many connections
```

**Diagnosis:**
```sql
-- Check current connections
SHOW PROCESSLIST;

-- Check max connections
SHOW VARIABLES LIKE 'max_connections';
-- Output: max_connections | 151

-- Check connection count
SHOW STATUS LIKE 'Threads_connected';
-- Output: Threads_connected | 151
```

**Root Cause:**
Application not closing database connections properly, leading to connection pool exhaustion.

**Solution:**
```php
<?php
// Before (Bad): Not closing connections
function getUsers() {
    $pdo = new PDO("mysql:host=localhost;dbname=test", "user", "pass");
    $stmt = $pdo->query("SELECT * FROM users");
    return $stmt->fetchAll();
    // Connection stays open!
}

// After (Good): Proper connection management
function getUsers() {
    $pdo = null;
    try {
        $pdo = new PDO("mysql:host=localhost;dbname=test", "user", "pass");
        $stmt = $pdo->query("SELECT * FROM users");
        $results = $stmt->fetchAll();
        return $results;
    } finally {
        $pdo = null;  // Explicitly close
    }
}

// Best: Use connection pooling
class Database {
    private static $instance = null;

    public static function getInstance() {
        if (self::$instance === null) {
            self::$instance = new PDO(
                "mysql:host=localhost;dbname=test",
                "user",
                "pass",
                [PDO::ATTR_PERSISTENT => true]
            );
        }
        return self::$instance;
    }
}
?>
```

**Prevention:**
- Use persistent connections
- Implement connection pooling
- Set appropriate timeout values
- Monitor connection usage

---

### Example 3: Permission Denied Error

**Error Message:**
```
PHP Warning: file_put_contents(/var/www/html/cache/data.json):
failed to open stream: Permission denied in /var/www/html/save.php on line 12
```

**Diagnosis:**
```bash
# Check file permissions
ls -la /var/www/html/cache/
# Output: drwxr-xr-x 2 root root 4096 Dec  1 10:00 cache

# Check web server user
ps aux | grep apache2 | head -1
# Output: www-data

# Check SELinux context
ls -Z /var/www/html/cache/
```

**Root Cause:**
Cache directory owned by root instead of web server user.

**Solution:**
```bash
# Fix ownership
chown -R www-data:www-data /var/www/html/cache

# Fix permissions
chmod -R 775 /var/www/html/cache

# If SELinux is enabled
chcon -R -t httpd_sys_rw_content_t /var/www/html/cache
# or
semanage fcontext -a -t httpd_sys_rw_content_t "/var/www/html/cache(/.*)?"
restorecon -R /var/www/html/cache
```

**Prevention:**
- Set correct ownership during deployment
- Use deployment scripts that set proper permissions
- Regular permission audits
- Document required permissions

---

### Example 4: Infinite Recursion

**Error Message:**
```
PHP Fatal error: Maximum function nesting level of '256' reached,
aborting! in /var/www/html/recursive.php on line 8
```

**Diagnosis:**
```php
<?php
// Code causing the error
function processData($data) {
    if (empty($data)) {
        return [];
    }
    // Bug: recursive call without proper exit condition
    return processData($data);
}
?>
```

**Solution:**
```php
<?php
// Fixed version with proper exit condition
function processData($data, $depth = 0, $maxDepth = 100) {
    if (empty($data) || $depth >= $maxDepth) {
        return [];
    }

    // Process data
    $result = doSomething($data);

    // Recursive call with depth tracking
    if (needsMoreProcessing($result)) {
        return processData($result, $depth + 1, $maxDepth);
    }

    return $result;
}
?>
```

**Prevention:**
- Always implement proper exit conditions
- Add depth/iteration limits
- Use iterative solutions when possible
- Implement safeguards against infinite loops

---

### Example 5: Slow Query Performance

**Error Message:**
```
Maximum execution time of 30 seconds exceeded in /var/www/html/report.php on line 67
```

**Diagnosis:**
```sql
-- Enable slow query log
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;

-- Check slow queries
SELECT * FROM mysql.slow_log ORDER BY start_time DESC LIMIT 10;

-- Analyze problematic query
EXPLAIN SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.status = 'pending'
ORDER BY o.created_at DESC;
```

**Root Cause:**
Missing indexes on frequently queried columns.

**Solution:**
```sql
-- Add appropriate indexes
CREATE INDEX idx_status ON orders(status);
CREATE INDEX idx_created_at ON orders(created_at);
CREATE INDEX idx_customer_id ON orders(customer_id);

-- Optimized query
SELECT
    o.id, o.total, c.name, c.email
FROM orders o
FORCE INDEX (idx_status, idx_created_at)
JOIN customers c ON o.customer_id = c.id
WHERE o.status = 'pending'
ORDER BY o.created_at DESC
LIMIT 100;
```

**Prevention:**
- Regular query performance analysis
- Proper indexing strategy
- Query result caching
- Pagination for large datasets

---

## Appendix: Useful Commands Reference

### Quick Diagnostic Commands

```bash
# System Resources
top -bn1 | head -20
free -h
df -h
uptime

# Web Server
systemctl status apache2
systemctl status nginx
apachectl configtest
nginx -t

# PHP
php -v
php -m
php -i | grep -i error

# Database
mysqladmin -u root -p status
psql -c "SELECT version();"

# Logs
tail -100 /var/log/apache2/error.log
tail -100 /var/log/nginx/error.log
journalctl -u apache2 -n 100
journalctl -u nginx -n 100

# Network
netstat -tuln
ss -tuln
lsof -i :80
lsof -i :443

# Process Management
ps aux | grep php-fpm
ps aux | grep apache2
pkill -9 php-fpm
```

---

## Summary

This comprehensive troubleshooting guide provides systematic approaches to identifying and resolving HTTP 500 Internal Server Errors. Remember:

1. **Stay Calm**: HTTP 500 errors are solvable with methodical debugging
2. **Check Logs First**: Error logs are your primary source of truth
3. **Use the Decision Tree**: Follow the diagnostic flowchart for systematic troubleshooting
4. **Document Everything**: Keep detailed records of issues and solutions
5. **Prevention is Key**: Implement monitoring, backups, and proper error handling
6. **Security Matters**: Never expose detailed errors in production
7. **Test Thoroughly**: Always test fixes in staging before production deployment

For ongoing issues or complex scenarios not covered in this guide, consider engaging specialized support or consulting with experienced system administrators.

---

**Document Version:** 1.0
**Last Updated:** December 2025
**Maintained By:** DevOps Team
**Questions/Updates:** devops@example.com