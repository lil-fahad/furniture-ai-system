# Security policy and urgent actions

## Urgent credential rotation

During consolidation, commit `69fa3f9e7e013861f3e7f0a5edb5d558df5c0838` in `lil-fahad/-furniture-ai` was found to disclose an Alibaba credential in its commit message.

Required actions:

1. Revoke/rotate that credential immediately in the Alibaba account or cloud console.
2. Review usage logs from the credential creation date onward.
3. Remove the secret from Git history using `git filter-repo` or the BFG Repo-Cleaner.
4. Force-push the cleaned history only after coordinating with every clone/fork.
5. Enable GitHub secret scanning and push protection.
6. Do not import the repository into this monorepo until a new audit passes.

The secret value is intentionally not repeated here.

## Repository rules

- Secrets belong only in environment variables or a managed secret store.
- Commit `.env.example`, never `.env`, credentials, tokens, or cloud key files.
- Every external source must be pinned to a full 40-character commit SHA.
- A blocked source must have no submodule path.
- Private repositories must remain private submodules and must not be vendored into this public repository.
- Model weights and datasets require checksums, source URLs, and license records.
- Do not load untrusted pickle/PyTorch model objects. Prefer `safetensors` or validated state dictionaries.
- Do not expose wildcard CORS with credentials in production.
- Replace all default JWT secrets and demo token endpoints before deployment.
- Product links, prices, availability, and reviews must be treated as unverified unless obtained from an authorized current API.

## Reporting

Open a private GitHub security advisory for vulnerabilities. Do not publish active credentials in issues, pull requests, commit messages, screenshots, or logs.
