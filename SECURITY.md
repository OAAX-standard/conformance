# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues, discussions or pull requests.

1. Use GitHub's private reporting: open the **Security** tab of this repository and choose **Report a vulnerability**. Only the maintainers can see the report.
2. If the button is not there (private repositories may not offer it), open an issue that says only "security contact needed", without details; a maintainer will reply with a private channel.
3. Include the affected version (`oaax-conformance --version`), the steps to reproduce, and the impact you expect.

## Scope

The suite is a **test tool, not a sandbox**: only check libraries and images you trust to run on your machine.

| What the suite runs | How |
| ------------------- | --- |
| the runtime library you give it | in a child process |
| the toolchain image you give it | in a container: every conversion runs as your user id with all Linux capabilities dropped, `no-new-privileges` and a process limit |

- `--docker-arg` can add back what an image really needs, and that is worth reporting to its vendor.
- Network access and writable mounts stay on by default, because a toolchain that needs them must be reported, not blocked. Separate conversions repeat the run:
  without network (TC-03), as another user id (TC-02, TC-22), with a read-only input (TC-04), a read-only root file system (TC-06) and an unwritable output directory (TC-38).
- Reports are written without the values of `--init-config` and without directory names.
