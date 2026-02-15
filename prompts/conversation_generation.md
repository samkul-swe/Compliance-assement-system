# Conversation Generation Prompt

## __ASK__

Generate a realistic customer service conversation between a collections agent and a customer.

Create a **{{TYPE}}** conversation with:
- 4-8 messages alternating between agent and customer
- Natural, realistic dialogue (not scripted or robotic)
- Specific dollar amount for the balance
- One clear customer situation (product loss, substandard service, or financial hardship)

Return ONLY valid JSON with the conversation.

## __CONTEXT__

You are creating training data for a compliance testing system. The conversation should feel realistic - like an actual phone call, chat, or email exchange between a collections agent and a customer who is behind on payments.

The collections team's goal is to engage customers while staying fully compliant with regulations (FDCPA) and earning high service ratings.

**Conversation Type Instructions:**
{{TYPE_INSTRUCTIONS}}

### Compliance Rules to Consider

{{RULES}}

### Customer Situations (pick ONE)

- **Product Loss**: Customer never received product, was charged for cancelled service, double charged, or received defective item
- **Substandard Service**: Premium features don't work, service is slow/unreliable, support unresponsive, app crashes frequently
- **Financial Hardship**: Lost job, medical emergency, divorce, unexpected bills, family emergency

## __CONSTRAINTS__

**Critical Requirements:**

1. **Natural language** - Write how people actually talk, including filler words, incomplete sentences if appropriate
2. **Realistic flow** - Conversations should feel like real customer service interactions
3. **Balanced length** - 4-8 messages total (2-4 exchanges)
4. **Valid JSON only** - No markdown code fences, no explanation, just the JSON
5. **Proper structure** - Each message must have "role", "text" fields
6. **Appropriate channel** - Match conversation style to channel (phone vs chat vs email)
7. **For non-compliant** - Make violations clear enough to detect, but natural (not forced)
8. **For vague** - Make genuinely ambiguous - could reasonably be interpreted either way

**JSON Structure Requirements:**

```json
{
  "messages": [
    {"role": "agent", "text": "..."},
    {"role": "customer", "text": "..."}
  ],
  "channel": "phone|chat|email",
  "customer_segment": "delinquent_30|delinquent_45|delinquent_60|delinquent_90"
}
```

## __EXAMPLE__

**Example 1: Compliant Conversation**

```json
{
  "messages": [
    {
      "role": "agent",
      "text": "Hi, I'm calling from the collections department about your account. I see you have a balance of $847. How can I help you today?"
    },
    {
      "role": "customer",
      "text": "Hi, yes. I've been meaning to call. I never actually received the product I was charged for."
    },
    {
      "role": "agent",
      "text": "I'm sorry to hear that. Let me look into this for you. Can you tell me more about what happened?"
    },
    {
      "role": "customer",
      "text": "I ordered a laptop two months ago but it never arrived. I've been charged though."
    },
    {
      "role": "agent",
      "text": "I understand. Let me escalate this to our fulfillment team right away. I'll also put a hold on collections while we investigate. You should hear back within 2 business days."
    },
    {
      "role": "customer",
      "text": "Thank you, I really appreciate your help."
    }
  ],
  "channel": "phone",
  "customer_segment": "delinquent_45"
}
```

**Example 2: Non-Compliant Conversation**

```json
{
  "messages": [
    {
      "role": "agent",
      "text": "You're 90 days past due on $2,340. We're going to sue you if this isn't paid by Friday."
    },
    {
      "role": "customer",
      "text": "I lost my job three weeks ago. I'm trying to find work but I need more time."
    },
    {
      "role": "agent",
      "text": "That's not our problem. You signed the agreement. Pay up or we're sending this to our lawyers."
    },
    {
      "role": "customer",
      "text": "Can we at least set up a payment plan?"
    },
    {
      "role": "agent",
      "text": "We need full payment immediately. Your credit is going to be destroyed if you don't handle this now."
    }
  ],
  "channel": "phone",
  "customer_segment": "delinquent_90"
}
```

**Example 3: Vague/Ambiguous Conversation**

```json
{
  "messages": [
    {
      "role": "agent",
      "text": "I'm calling about your account balance of $1,155. We really need to get this resolved soon."
    },
    {
      "role": "customer",
      "text": "I understand. I've had some unexpected medical bills this month."
    },
    {
      "role": "agent",
      "text": "I see. Well, time is of the essence here. The longer this goes on, the more difficult it becomes."
    },
    {
      "role": "customer",
      "text": "What are my options?"
    },
    {
      "role": "agent",
      "text": "I strongly encourage you to make a payment today. This situation is becoming more serious and we're running out of options."
    },
    {
      "role": "customer",
      "text": "I'll see what I can do."
    }
  ],
  "channel": "chat",
  "customer_segment": "delinquent_60"
}
```

---

**Remember:** The goal is realistic conversations that test the compliance system's ability to handle clear cases AND ambiguous edge cases. Make the language natural, not forced.