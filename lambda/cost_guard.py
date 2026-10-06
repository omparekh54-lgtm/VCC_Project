"""Stop this demo before its credit allowance or 21-day lifetime is exhausted."""
import os
import time
import boto3

def lambda_handler(event, context):
    region = os.environ['AWS_REGION']
    reason = None
    balance = None
    used = None
    try:
        plan = boto3.client('freetier', region_name='us-east-1').get_account_plan_state()
        balance = float(plan['accountPlanRemainingCredits']['amount'])
        activities = []
        params = {}
        free_tier = boto3.client('freetier', region_name='us-east-1')
        while True:
            page = free_tier.list_account_activities(**params)
            activities.extend(page.get('activities', []))
            if not page.get('nextToken'):
                break
            params['nextToken'] = page['nextToken']
        granted = float(os.environ['INITIAL_FREE_CREDITS']) + sum(
            float(a['reward']['credit']['amount']) for a in activities
            if a.get('status') == 'COMPLETED' and
            a.get('reward', {}).get('credit', {}).get('unit') == 'USD')
        used = max(0.0, granted - balance)
        if plan.get('accountPlanType') != 'FREE' or plan.get('accountPlanStatus') != 'ACTIVE':
            reason = 'Free account plan is no longer active'
        elif used >= float(os.environ['MAX_CREDIT_USAGE']):
            reason = f'Credit consumption reached {used}'
    except Exception as exc:
        # Fail closed: protecting the credit allowance takes priority over uptime.
        reason = f'Unable to verify credit protection: {exc}'
    if time.time() >= float(os.environ['STOP_AT_EPOCH']):
        reason = '21-day demo lifetime ended'
    if not reason:
        return {'status': 'running', 'remaining_credits': balance, 'credits_used': used}
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
