import os.path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/tasks"]
class GoogleTask:

  def __init__(self):
    """Shows basic usage of the Tasks API.
    Prints the title and ID of the first 10 task lists.
    """
    creds = None
    if os.path.exists("token.json"):
      creds = Credentials.from_authorized_user_file("token.json", SCOPES)
      # Token created with different scopes cannot be refreshed for new scopes.
      if not creds.has_scopes(SCOPES):
        creds = None
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
        try:
          creds.refresh(Request())
        except RefreshError:
          creds = None
      else:
        creds = None
      if not creds:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open("token.json", "w") as token:
        token.write(creds.to_json())
    self.__service = build("tasks", "v1", credentials=creds)

  def getTasks(self) -> None:
    try:
      result = self.__service.tasks().list(tasklist="@default").execute()
      items = result.get("items", [])

      if not items:
        print("No task lists found.")
        return

      return items
    except HttpError as error:
      print(f"An error occurred: {error}")

  def createTask(
      self,
      title: str,
      notes: str | None = None,
      due: str | None = None,
      status: str | None = None,
      tasklist: str = "@default",
  ) -> dict | None:
    try:
      task_body = {"title": title}
      if notes is not None:
        task_body["notes"] = notes
      if due is not None:
        task_body["due"] = due
      if status is not None:
        task_body["status"] = status

      created_task = (
          self.__service.tasks().insert(tasklist=tasklist, body=task_body).execute()
      )
      print(f"Task created: {created_task['title']} ({created_task['id']})")
      return created_task
    except HttpError as error:
      print(f"An error occurred while creating task '{title}': {error}")
      return None

  def updateTask(
      self,
      task_id: str,
      title: str | None = None,
      notes: str | None = None,
      status: str | None = None,
      due: str | None = None,
      tasklist: str = "@default",
  ) -> dict | None:
    try:
      task = self.__service.tasks().get(tasklist=tasklist, task=task_id).execute()

      if title is not None:
        task["title"] = title
      if notes is not None:
        task["notes"] = notes
      if status is not None:
        task["status"] = status
      if due is not None:
        task["due"] = due

      updated_task = (
          self.__service.tasks()
          .update(tasklist=tasklist, task=task_id, body=task)
          .execute()
      )
      print(f"Task updated: {updated_task['title']} ({updated_task['id']})")
      return updated_task
    except HttpError as error:
      print(f"An error occurred while updating task {task_id}: {error}")
      return None

  def deleteTask(self, task_id: str, tasklist: str = "@default") -> bool:
    try:
      self.__service.tasks().delete(tasklist=tasklist, task=task_id).execute()
      print(f"Task deleted: {task_id}")
      return True
    except HttpError as error:
      print(f"An error occurred while deleting task {task_id}: {error}")
      return False

# if __name__ == "__main__":
#   task = GoogleTask()
#   task.getTasks()

# if __name__ == "__main__":
#   task = GoogleTask()
#   task.createTask(
#       title="Pagar boleto",
#       notes="Vencimento dia 10",
#       due="2026-02-25T12:00:00.000Z"
#   )
#   task.getTasks()

# if __name__ == "__main__":
#   gt = GoogleTask()
#   tasks = gt.getTasks()
#   if tasks:
#       task_id = tasks[0]["id"]
#       gt.updateTask(task_id=task_id, title="Título editado")

# if __name__ == "__main__":
#   gt = GoogleTask()
#   tasks = gt.getTasks()
#   if tasks:
#       task_id = tasks[0]["id"]
#       gt.deleteTask(task_id=task_id)
