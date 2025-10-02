#!/usr/bin/env python3
"""
Setup IoT device keys + CA and dummy data inside Docker.
- Generates Ed25519 device key pair if not exists
- Generates self-signed CA and a device certificate signed by the CA
- Creates signed dummy IoT readings (count = NUM_READINGS from env)
"""

import os, json
from pathlib import Path
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509 import Name, NameAttribute
from cryptography.x509.oid import NameOID

# --- Anzahl Readings aus ENV übernehmen ---
NUM_READINGS = int(os.environ.get("NUM_READINGS", "1000"))

# --- Paths ---
base_dir = Path(__file__).resolve().parent.parent
keys_dir = base_dir / "data" / "device_keys"
raw_dir = base_dir / "data" / "raw"

keys_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)

priv_path = keys_dir / "device1_private.pem"
pub_path = keys_dir / "device1_public.pem"
dev_cert_path = keys_dir / "device1_cert.pem"
ca_key_path = keys_dir / "ca_private.pem"
ca_cert_path = keys_dir / "ca_cert.pem"
data_path = raw_dir / "iot_readings_1_month.json"

# --- Keys erzeugen oder laden ---
if priv_path.exists() and pub_path.exists():
    with open(priv_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    public_key = private_key.public_key()
    print("[INFO] Device keys reused.")
else:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    print("[INFO] Device keys generated.")

# --- CA (issuer) Schlüssel und Zertifikat ---
if ca_key_path.exists() and ca_cert_path.exists():
    with open(ca_key_path, "rb") as f:
        ca_private_key = serialization.load_pem_private_key(f.read(), password=None)
    with open(ca_cert_path, "rb") as f:
        ca_cert = x509.load_pem_x509_certificate(f.read())
    print("[INFO] CA reused.")
else:
    ca_private_key = ed25519.Ed25519PrivateKey.generate()
    ca_subject = Name([
        NameAttribute(NameOID.COUNTRY_NAME, u"DE"),
        NameAttribute(NameOID.ORGANIZATION_NAME, u"Local Test CA"),
        NameAttribute(NameOID.COMMON_NAME, u"Local Test Root CA"),
    ])
    ca_builder = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    )
    ca_cert = ca_builder.sign(private_key=ca_private_key, algorithm=None)
    with open(ca_key_path, "wb") as f:
        f.write(ca_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(ca_cert_path, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
    print("[INFO] CA generated.")

# --- Device-Zertifikat signiert von CA ---
if dev_cert_path.exists():
    with open(dev_cert_path, "rb") as f:
        device_cert = x509.load_pem_x509_certificate(f.read())
    print("[INFO] Device certificate reused.")
else:
    dev_subject = Name([
        NameAttribute(NameOID.COUNTRY_NAME, u"DE"),
        NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, u"IoT Device"),
        NameAttribute(NameOID.COMMON_NAME, u"device1"),
    ])
    dev_builder = (
        x509.CertificateBuilder()
        .subject_name(dev_subject)
        .issuer_name(ca_cert.subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=365 * 3))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    )
    device_cert = dev_builder.sign(private_key=ca_private_key, algorithm=None)
    with open(dev_cert_path, "wb") as f:
        f.write(device_cert.public_bytes(serialization.Encoding.PEM))
    print("[INFO] Device certificate generated and signed by CA.")

# --- Dummy-Readings erzeugen ---
readings = []
for i in range(NUM_READINGS):
    val = 20.0 + (i * 0.01)  # Dummy-Sensorwert
    data = {"value": val}
    msg = json.dumps(data, sort_keys=True).encode("utf-8")
    sig = private_key.sign(msg)
    readings.append({
        "data": data,
        "signature": sig.hex()
    })

with open(data_path, "w") as f:
    json.dump(readings, f, indent=2)

print(f"[INFO] {NUM_READINGS} signed IoT readings created at {data_path}")
