import json
import time
import urllib.request
import urllib.error
import sys

def main():
    print("Loading test workflow from samples/flux2-klein-9b-text-to-image-api.json...")
    try:
        with open("samples/flux2-klein-9b-text-to-image-api.json", "r") as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"Error loading workflow file: {e}")
        sys.exit(1)

    payload = {
        "workflow": workflow
    }

    # Submit job to local proxy
    print("Submitting job to local proxy (http://127.0.0.1:3000/api/jobs)...")
    req = urllib.request.Request(
        "http://127.0.0.1:3000/api/jobs",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as res:
            job = json.loads(res.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Submission failed: {e.code} - {e.read().decode()}")
        sys.exit(1)
    except Exception as e:
        print(f"Submission failed: {e}")
        sys.exit(1)
        
    job_id = job.get("id")
    status = job.get("status")
    print(f"Job submitted successfully. ID: {job_id}, Initial Status: {status}")

    # Poll status
    print("Polling job status (interval: 5s)...")
    while True:
        status_req = urllib.request.Request(f"http://127.0.0.1:3000/api/jobs/{job_id}")
        try:
            with urllib.request.urlopen(status_req) as res:
                job_status = json.loads(res.read().decode())
        except Exception as e:
            print(f"Error polling status: {e}")
            time.sleep(5)
            continue
            
        status = job_status.get("status")
        delay = job_status.get("delayTime", 0)
        execution = job_status.get("executionTime", 0)
        
        print(f"Status: {status} | Delay: {delay}s | Execution: {execution}s")
        
        if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"]:
            break
            
        time.sleep(5)

    if status == "COMPLETED":
        print("\nSUCCESS! Generation job completed successfully.")
        print(f"Saved images/outputs metadata: {job_status.get('saved', [])}")
    else:
        print(f"\nFAILURE! Job ended with status: {status}")
        print(f"Error details: {job_status.get('error') or job_status.get('message')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
