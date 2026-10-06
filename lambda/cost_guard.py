"""Stop this demo before its credit allowance or 21-day lifetime is exhausted."""
import os
import time
import boto3

def lambda_handler(event, context):
    region = os.environ['AWS_REGION']
    reason = None
    balance = None
    try:
        plan = boto3.client('freetier', region_name='us-east-1').get_account_plan_state()
        balance = float(plan['accountPlanRemainingCredits']['amount'])
        if plan.get('accountPlanType') != 'FREE' or plan.get('accountPlanStatus') != 'ACTIVE':
            reason = 'Free account plan is no longer active'
        elif balance <= float(os.environ['MIN_REMAINING_CREDITS']):
            reason = f'Credit balance reached {balance}'
    except Exception as exc:
        # Fail closed: protecting the credit allowance takes priority over uptime.
        reason = f'Unable to verify credit protection: {exc}'
    if time.time() >= float(os.environ['STOP_AT_EPOCH']):
        reason = '21-day demo lifetime ended'
    if not reason:
        return {'status': 'running', 'remaining_credits': balance}
    events = boto3.client('events', region_name=region)
    events.disable_rule(Name=os.environ['SCALING_RULE'])
    boto3.client('autoscaling', region_name=region).update_auto_scaling_group(
        AutoScalingGroupName=os.environ['ASG_NAME'], MinSize=0, MaxSize=0, DesiredCapacity=0)
    sm = boto3.client('sagemaker', region_name=region)
    try:
        sm.delete_endpoint(EndpointName=os.environ['ENDPOINT_NAME'])
    except sm.exceptions.ClientError as exc:
        if 'Could not find' not in str(exc) and 'does not exist' not in str(exc):
            raise
    for page in sm.get_paginator('list_training_jobs').paginate(NameContains='infra-watchdog', StatusEquals='InProgress'):
        for job in page.get('TrainingJobSummaries', []):
            if job['TrainingJobName'].startswith('infra-watchdog'):
                sm.stop_training_job(TrainingJobName=job['TrainingJobName'])
    print({'shutdown_reason': reason, 'remaining_credits': balance})
    return {'status': 'stopped', 'reason': reason, 'remaining_credits': balance}
