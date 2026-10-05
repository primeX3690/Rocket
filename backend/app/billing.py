"""
billing.py

Billing integration skeleton. This is DELIBERATELY a skeleton, not a
working Stripe/Razorpay integration: a real integration needs a real
merchant account, real API keys, and a publicly reachable webhook URL
(so Stripe/Razorpay can call back) -- none of which exist inside this
development/demo environment, and faking them would mean shipping code
that LOOKS like it processes payments but doesn't, which is worse than
clearly marking it as not-yet-wired.

What's real here: the PLAN/QUOTA model (server.py's PLAN_QUOTAS,
enforced on every simulation call), and the exact shape of the two
functions a real integration needs to fill in.

TO GO LIVE (either provider):
  Stripe:
    1. pip install stripe; set STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
    2. create_checkout_session -> stripe.checkout.Session.create(...)
       with success_url/cancel_url pointing back at your frontend
    3. handle_webhook_event -> verify the signature
       (stripe.Webhook.construct_event), then on
       'checkout.session.completed' call db logic to set the org's
       plan (add an update_organization_plan() function to db.py)
  Razorpay (often simpler for an India-only customer base, INR-native):
    1. pip install razorpay; set RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET
    2. create_checkout_session -> client.order.create(...)
    3. handle_webhook_event -> verify the signature
       (razorpay's utility.verify_webhook_signature), then update the
       org's plan the same way
"""

import os

PLAN_PRICES_INR = {"pro": 4999, "enterprise": None}   # None = "contact us" pricing


def create_checkout_session(org_id: int, requested_plan: str) -> dict:
    if requested_plan not in PLAN_PRICES_INR:
        return {"error": f"Unknown plan '{requested_plan}'. Valid plans: {list(PLAN_PRICES_INR)}"}

    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    razorpay_key = os.environ.get("RAZORPAY_KEY_ID")
    if not stripe_key and not razorpay_key:
        return {
            "status": "not_configured",
            "message": (
                "Billing is not yet connected to a payment provider. Set "
                "STRIPE_SECRET_KEY (or RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET) "
                "as environment variables and implement the provider call "
                "in billing.py's create_checkout_session() — see that "
                "function's docstring for the exact steps."
            ),
            "requested_plan": requested_plan, "price_inr": PLAN_PRICES_INR[requested_plan],
            "org_id": org_id,
        }
    # Real provider call goes here once keys are configured.
    return {"status": "not_implemented",
           "message": "Payment keys were found but the provider call itself is not yet implemented. "
                      "See billing.py's module docstring for the exact Stripe/Razorpay calls to add."}


def handle_webhook_event(payload: dict) -> dict:
    return {"status": "not_configured",
           "message": "Webhook handling is not yet implemented — see billing.py's module docstring."}
