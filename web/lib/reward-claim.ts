const REWARD_EMAIL = "jianouyang001@gmail.com";

type RewardClaim = {
  claimantEmail: string;
  gameId: string;
  name: string;
  note: string;
};

export function rewardClaimMailto(claim: RewardClaim) {
  const subject = `Shallow Red reward claim — ${claim.gameId.slice(0, 8)}`;
  const body = [
    `Name: ${claim.name}`,
    `Reply-to email: ${claim.claimantEmail}`,
    `Game ID: ${claim.gameId}`,
    "",
    "Message:",
    claim.note || "(No message provided.)",
  ].join("\n");

  return `mailto:${REWARD_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}
