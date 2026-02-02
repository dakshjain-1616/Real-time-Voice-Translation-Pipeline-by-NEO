# Check memory before start
free -h || vm_stat
# Run verification script
python3 src/verify_realtime.py
# Check outputs
ls -lh outputs/realtime_test_*