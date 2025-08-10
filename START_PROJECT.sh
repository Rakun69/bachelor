#!/bin/bash

# Complete Project Execution Script
# Generates all visualizations and results for the Bachelor Thesis

echo "🚀 Starting IoT ZK-SNARK Evaluation System"
echo "=========================================="

# Activate virtual environment
source iot_zk_env/bin/activate

echo "✅ Virtual environment activated"

# Step 1: Run Demo (Quick test + some visualizations)
echo ""
echo "📊 Step 1: Running Demo (generates basic visualizations)..."
python3 demo.py

echo ""
echo "🔍 Step 2: Running Crossover Analysis (generates theoretical analysis)..."
python3 src/analysis/crossover_point_analyzer.py

echo ""
echo "📈 Step 3: Checking generated files..."
echo "Generated visualizations:"
ls -la data/visualizations/*.png 2>/dev/null || echo "No PNG files found yet"
ls -la data/visualizations/*.txt 2>/dev/null || echo "No TXT files found yet"

echo ""
echo "🎯 Step 4: Running complete evaluation..."
# Option A: Full evaluation (takes longer)
# ./run_evaluation.sh --phase all

# Option B: Individual phases
echo "Running individual evaluation phases..."
# Setup environment and dependencies
./run_evaluation.sh --setup
# Generate data, compile, analyze and visualize
./run_evaluation.sh --phase data
./run_evaluation.sh --phase compile
./run_evaluation.sh --phase analyze
./run_evaluation.sh --phase visualize

echo ""
echo "✅ Project execution completed!"
echo ""
echo "📊 Generated Results:"
echo "===================="
echo "📁 Visualizations: data/visualizations/"
echo "📁 Reports: data/benchmarks/"
echo "📁 LaTeX Thesis: thesis_sections/bachelorarbeit.pdf"
echo "📁 Analysis Data: data/crossover_analysis_report.json"

echo ""
echo "🎓 Your thesis project is ready!"