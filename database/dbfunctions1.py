import datetime
import sqlite3
from datetime import date

DATABASE_PATH = "E:/Project25/J_Blosmia1/database/database1.db"

# ================= ADMIN FUNCTIONS =================


# Authenticates admin login credentials.
def admin(username, password):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "select * from admin where uname=? And password =?", (username, password)
    )
    res = cur.fetchall()
    print(res)
    con.close()
    return res


# Fetches all users from the admin table.
def getusers():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT  Id,email_id,type,uname FROM admin")
    res = cur.fetchall()
    con.close()
    return res


# Adds a new user into the admin table.
def addusers(data):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "INSERT INTO admin VALUES((SELECT COUNT(Id) FROM admin)+1,?,?,?,?)", data
        )
        con.commit()
        con.close()
        return True
    except Exception as e:
        return False


# Deletes a user by username.
def deleteUser(username):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("DELETE FROM admin WHERE uname=?", [username])
    con.commit()
    con.close()


# Updates user details (role/type).
def editUser(data):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("UPDATE admin SET type=? WHERE username=?", data)
    con.commit()
    con.close()


# ================= PATIENT FUNCTIONS =================


# Returns the next available patient number (ID generator).
def getPatientsNumber():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM patient_details")
    count = cur.fetchall()[0][0]
    con.close()
    return 100 + count + 1


# Fetches details of all patients.
def getpatientdetails():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from patient_details")
    pdetails = cur.fetchall()
    con.close()
    return pdetails


# Fetches all details of a specific patient.
def getpatientdetail(patient_id):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from patient_details where patient_id=?", [patient_id])
    pdetails = cur.fetchall()
    con.close()
    return pdetails


# Fetches only basic details (ID, name, DOB, gender) of a patient.
def getpatient_details(patient_id):
    img_id = 1
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "select patient_id,patient_name,DOB,gender from patient_details where patient_id=?",
        [patient_id],
    )
    detail = cur.fetchall()
    con.close()
    return detail


# Inserts a new patient record and links it to a doctor.
def addnewpatients(data, doc_id):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        con.execute("BEGIN TRANSACTION")
        cur.execute(
            "insert into patient_details(patient_name,emailid,mobile_no,gender,DOB,address,Blood_grp)values(?,?,?,?,?,?,?)",
            (data),
        )
        last_inserted_id = cur.lastrowid
        print("pid is", last_inserted_id)
        # TO DO: Add doc id and patient id to patient doc table
        cur.execute(
            "insert into patient_Doctor(doc_id,patient_id)values(?,?)",
            (doc_id, last_inserted_id),
        )
        con.commit()
        return last_inserted_id
    except Exception as e:
        print("exception+", e)
        con.rollback()
        return -1
    finally:
        con.close()


# ================= DOCTOR FUNCTIONS =================


# Fetches details of all doctors.
def getdoctor_details():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from doctor_details")
    ddeatils = cur.fetchall()
    con.close()
    return ddeatils


# Inserts a new doctor record.
def addnewdoc_details(data):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        print(data)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "insert into doctor_details(doc_name,designation,department,email_id,phn_no,hospital_name)values(?,?,?,?,?,?)",
            data,
        )
        con.commit()
        return True
    except Exception as e:
        print(e)
        return False


# Retrieves the last doctor IDs from the database.
def getlast_docid():
    con = sqlite3.connect(DATABASE_PATH)
    # con.row_factory=sqlite3.Row
    cur = con.cursor()
    cur.execute("Select doc_id from doctor_details")
    doc_ids = cur.fetchall()
    con.close()
    return doc_ids


# ================= REPORT & IMAGE FUNCTIONS =================


# Creates a new report for a patient.
def report(data):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        # count
        cur.execute("insert into report(patient_id, date_and_time) values(?, ?)", data)
        last_inserted_id = cur.lastrowid
        con.commit()
        return last_inserted_id
    except Exception as e:
        print(e)
        return False


# Saves uploaded images to image_data table for a report.
def addimage(rep_id, filename, image_blob, folder_name):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            'INSERT INTO image_data(report_id, img, foldername) VALUES (?, ?, ?)',
            (rep_id, image_blob, folder_name)
        )
        last_inserted_id = cur.lastrowid
        con.commit()
        return last_inserted_id
    except Exception as e:
        print(e)
        return False
    finally:
        con.close()


# Countes stored images.
def imagecount():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT image_id FROM image_data ORDER BY image_id DESC LIMIT 1")
    count = cur.fetchall()[0][0]
    con.close()
    return count


# Fetches RBC analysis data for a patient.
def getrbcData(patient_id):  # have to work on
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        # cur.execute("SELECT  data_id,create_date FROM patient_data WHERE patient_id=?",[patient_id])
        # select imgid from img table where rep=rep in report where patientid=
        print(type(patient_id))
        cur.execute(
            "select report_id,date_and_time from report where patient_id=?",
            [patient_id],
        )
        res = cur.fetchall()
        con.close()
        return res
    except Exception as e:
        print("error is", e)
        return False


# ================= BLOB FUNCTIONS =================


# Insert blob (cell image) into main BLOB table and update report cell counts.
def types(img_id, report_id, image, cell_id):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute(
            "INSERT INTO BLOB_TAB(img_id, blob_image, CellsId, date_and_time, trained) VALUES (?, ?, ?, ?, 0)",
            (img_id, image, int(cell_id), timestamp),
        )
        # Get cell type name from CellsType using CellsId
        cur.execute("SELECT CellsName FROM CellsType WHERE CellsId = ?", (cell_id,))
        col = cur.fetchone()
        if col:
            column_name = col[0]  # e.g., 'Neutrophil' or 'double rbc'
            # Normalize column name if needed (replace space with underscore)
            column_name = column_name.replace(" ", "_")
            # Now update ReportResult
            update_query = f"UPDATE ReportResult SET [{column_name}] = [{column_name}] + 1 WHERE report_id = ?"
            cur.execute(update_query, (report_id,))
        # timeStamp=str(datetime.date.today()).replace(":","-")
        data = [image, types]
        # select cellid from celltype where cellname=type
        # select imgid from imagedata where(won't work bcause there r more than ond image with same name)
        con.commit()
        con.close()
    except Exception as e:
        print("from here", e)
        return False


# Remove blobs for a given report ID.
def clear_existing_blobs(img_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM BLOB_TAB WHERE img_id = ?", (img_id,))
        conn.commit()
        print(f"Cleared existing blobs for img_id {img_id}")
    except Exception as e:
        print(f"Failed to clear blobs: {e}")
    finally:
        conn.close()


# Modifies blob classification.
def editreport(blob_id, type):
    types = [
        "Acanthocytes",
        "basophil",
        "eosinophil",
        "Lymphocyte",
        "Monocyte",
        "Neutrophil",
        "Normal RBC",
        "Stomachocytes",
        "Tear Drop",
        "Uncategorized",
        "single rbc",
        "double rbc",
        "triple rbc",
    ]
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("UPDATE BLOB_TAB set CellsId=? WHERE blob_id=?", (type, blob_id))
    res = cur.fetchall()
    con.commit()
    return True


# Fetches blob images for displaying in reports.
def get_blob_data_to_display(rep_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "select bt.blob_id,bt.blob_image,ct.CellsName,bt.CellsId from BLOB_TAB bt inner join image_data id on  bt.img_id=id.image_id INNER join  CellsType ct on bt.CellsId=ct.CellsId and id.report_id=?",
        [rep_id],
    )
    res = cur.fetchall()
    conn.close()
    return res


# ================= REPORT RESULT FUNCTIONS =================


# Insert initial cell counts into ReportResult table.
def add_count_in_reportResult(cell_count_data):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "insert into ReportResult(report_id,basophil,double_rbc,eosinophil,Lymphocyte,Monocyte,Neutrophil,single_rbc,triple_rbc) values(?,?,?,?,?,?,?,?,?)",
            cell_count_data,
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("addcount", e)
        return False


# Computestotal cells in a report.
def get_total_count(rep_id):
    conn = sqlite3.connect(DATABASE_PATH)
    # conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT  COUNT(bt.CellsId) FROM CellsType ct left JOIN (BLOB_TAB bt  INNER JOIN image_data img ON bt.img_id = img.image_id AND img.report_id=?)ON ct.CellsId = bt.CellsId GROUP BY ct.CellsId",
        [rep_id],
    )
    res = cur.fetchall()
    conn.close()
    return res


# Update existing cell counts in ReportResult.
def update_report_result(cell_count_data):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "update ReportResult set basophil=?,double_rbc=?,eosinophil=?,Lymphocyte=?,Monocyte=?,Neutrophil=?,single_rbc=?,triple_rbc=? where report_id=?",
            cell_count_data,
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("addcount", e)
        return False


# Fetches counts of cells for a report.
def get_report_result_data(rep_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "Select basophil,double_rbc,eosinophil,Lymphocyte,Monocyte,Neutrophil,single_rbc,triple_rbc from ReportResult where report_id=?",
        [rep_id],
    )
    res = cur.fetchall()
    conn.close()
    return res


# Retrieve report result data for PDF generation.
def pdf(patient_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "Select basophil,double_rbc,eosinophil,Lymphocyte,Monocyte,Neutrophil,single_rbc,triple_rbc from ReportResult RR inner JOIN report rp on RR.report_id = rp.report_id where rp.patient_id=?",
        [patient_id],
    )
    invoice = cursor.fetchall()
    conn.close()
    return invoice


# ================= TRAINING FUNCTIONS =================


# Fetch retraining images and labels from trainedimages.
def get_blob_retrain():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select trainedimg,CellsId from trainedimages")
    res = cur.fetchall()
    conn.close()
    return res


# Insert a new image into training table for a given cell type.
def inserttested(binary_data, cellsid):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "INSERT INTO trainedimages (trainedimg,CellsId)  values(?,?)",
            (binary_data, cellsid),
        )
        con.commit()
    except Exception as e:
        print("from here", e)
        return False


# Fetch untrained blob images and their labels.
def get_blob_to_trained():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("select CellsId,blob_image from BLOB_TAB where trained=0")
    res = cur.fetchall()
    conn.close()
    return res


# Mark all untrained blobs as trained after model update.
def update_blob_after_train():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("update BLOB_TAB set trained=1 where trained=0")
    conn.commit()
    res = cur.fetchall()
    conn.close()
    return True


# ================= UNUSED / HELPER FUNCTIONS =================

# Insert blob (cell image) into WBC BLOB table for training/classification.
def types_wbc(img_id, report_id, image, cell_id):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cur.execute(
            """
            INSERT INTO BLOB_WBC_TAB(img_id, blob_image, CellsId, date_and_time, trained)
            VALUES (?, ?, ?, ?, 0)
        """,
            (img_id, image, int(cell_id), timestamp),
        )

        con.commit()
        con.close()
    except Exception as e:
        print("WBC insert error:", e)
        return False


# Fetch all reports/images linked to a patient.
def getreport(patient_id):  # have to change
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    # cur.execute("SELECT patient_id,report_id,image_id,date_and_time FROM image_data WHERE patient_id=?",[patient_id])
    # write a join on image data and report on repid=repid where patient id=patient_id
    cur.execute(
        "select r.patient_id,im.report_id,im.image_id,im.foldername from image_data im inner join report r on im.report_id=r.report_id where r.patient_id=?",
        [patient_id],
    )
    res = cur.fetchall()
    con.close()
    return res


# Fetches a blob by patient & image ID.
def getblob(patient_id, img_id):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT blob_id,type from BLOB_TAB WHERE patient_id=? AND img_id=?",
        [patient_id, img_id],
    )
    res = cur.fetchall()
    return res


# Fetches blob ID linked to patient & image
def getblobid(patient_id, img_id):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT blob_id from BLOB_TAB WHERE patient_id=? AND img_id=?",
        [patient_id, img_id],
    )
    res = cur.fetchall()[0][0]
    return res


# Fetches combined patient + report details.
def getAllPatientDetails(patient_id, d):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT r.patient_id,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Acanthocytes' AND b.patient_id = r.patient_id) AS Acanthocytes_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'basophil' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS basophil_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'eosinophil' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS eosinophil_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Lymphocyte' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS Lymphocyte_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Monocyte' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS Monocyte_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Neutrophil' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS Neutrophil_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'normal rbc' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS normal_rbc_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Stomachocytes' AND b.patient_id = r.patient_id) AS Stomachocytes_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Tear Drop' AND b.patient_id = r.patient_id) AS Tear_Drop_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'Uncategorized' AND b.patient_id = r.patient_id) AS Uncategorized_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'single rbc' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS single_rbc_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'double rbc' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS double_rbc_count,(SELECT COUNT(*) FROM BLOB_TAB b WHERE b.type = 'triple rbc' AND b.patient_id = r.patient_id AND b.date_and_time = i.date_and_time) AS triple_rbc_count FROM report r JOIN image_data i ON r.patient_id = i.patient_id WHERE  r.patient_id = ? AND i.date_and_time = ? GROUP BY r.patient_id, i.date_and_time;",
        [patient_id, d],
    )
    res = cur.fetchall()
    con.close()
    return res


# Inserts images into training table.
def traintab(image, cell_id):
    try:
        con = sqlite3.connect(DATABASE_PATH)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        # timeStamp=str(datetime.date.today()).replace(":","-")
        data = [image, types]
        # print(img_id)
        cur.execute(
            "insert into trainedimages (trainedimg,CellsId) values(?,?)",
            (image, cell_id),
        )
        # select cellid from celltype where cellname=type
        # traintab is db is not used
        # select imgid from imagedata where(won't work bcause there r more than ond image with same name)
        con.commit()
    except Exception as e:
        print("from here", e)
        return False


# Not used; intended for image processing input.
def getbsi(uploaded_images):
    try:
        imgid = getsmearcount()
        idate = date.today()
        print(idate)
        for file in uploaded_images:
            image_data = file.read()
            idata = [imgid, image_data, idate]
            con = sqlite3.connect(DATABASE_PATH)
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "insert into image_table(img_id,img,date_of_report) Values(?,?,?)",
                idata,
            )
            con.commit()
            imgid += 1
        return True
    except Exception as e:
        print(e)
        return False


# Not used; intended to count smear samples.
def getsmearcount():
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM image_table")
    count = cur.fetchall()[0][0]
    con.close()
    return count + 1


# Not used; intended to count blob types.
def gettypescount(blob_id):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    types = []
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("basophil", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("eosinophil", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Lymphocyte", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Monocyte", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Neutrophil", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Acanthocytes", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Normal RBC", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Stomachocytes", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("Tear Drop", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("single rbc", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("double rbc", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and img_id=? ",
        ("triple rbc", blob_id),
    )
    types.append(cur.fetchall()[0][0])
    con.close()
    return types


# Not used; intended for image-based blob counts.
def gettypescounts(blob_id, imageid):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    types = []
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("basophil", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("eosinophil", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Lymphocyte", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Monocyte", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Neutrophil", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Acanthocytes", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Normal RBC", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Stomachocytes", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("Tear Drop", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("single rbc", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("double rbc", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    cur.execute(
        "SELECT  count(blob_id) FROM BLOB_TAB WHERE type=? and patient_id=? and img_id=?",
        ("triple rbc", blob_id, imageid),
    )
    types.append(cur.fetchall()[0][0])
    con.close()
    return types


# Not used; intended for folder path operations.
def folder_p(rep_id):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT foldername FROM image_data where report_id=?", [rep_id])
    folder_name = cur.fetchall()[0][0]
    con.close()
    return folder_name


# Not used; intended to fetch image IDs from a folder.
def image_id(folder):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT image_id from image_data where foldername=?", [folder])
    res = cur.fetchall()
    con.close()
    return res


# Not used; intended to store RBC file path for a patient.
def addrbcFilePath(patient_id, path):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    data = [patient_id, path, date.today()]
    cur.execute(
        "INSERT INTO patient_data VALUES((SELECT COUNT(data_id) FROM patient_data)+1,?,?,?)",
        data,
    )
    con.commit()
    return True


# Commented; for ROI blob handling.
def blob(ROI):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    data = [1, ROI, date.today()]
    cur.execute("INSERT INTO blob_table VALUES(?,?,?)", data)
    con.commit()
    return True


# Commented; older version of patient details retrieval.
def getAllPatientDetails(patient_id):
    con = sqlite3.connect(DATABASE_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        "SELECT r.patient_id,SUM(CASE WHEN b.type = 'Acanthocytes' THEN 1 ELSE 0 END) AS Acanthocytes_count,SUM(CASE WHEN b.type = 'basophil' THEN 1 ELSE 0 END) AS basophil_count,SUM(CASE WHEN b.type = 'eosinophil' THEN 1 ELSE 0 END) AS eosinophil_count,SUM(CASE WHEN b.type = 'Lymphocyte' THEN 1 ELSE 0 END) AS Lymphocyte_count,SUM(CASE WHEN b.type = 'Monocyte' THEN 1 ELSE 0 END) AS Monocyte_count,SUM(CASE WHEN b.type = 'Neutrophil' THEN 1 ELSE 0 END) AS Neutrophil_count,SUM(CASE WHEN b.type = 'normal rbc' THEN 1 ELSE 0 END) AS normal_rbc_count,SUM(CASE WHEN b.type = 'Stomachocytes' THEN 1 ELSE 0 END) AS Stomachocytes_count,SUM(CASE WHEN b.type = 'Tear Drop' THEN 1 ELSE 0 END) AS Tear_Drop_count,SUM(CASE WHEN b.type = 'Uncategorized' THEN 1 ELSE 0 END) AS Uncategorized_count,COUNT(CASE WHEN b.type = 'single rbc' THEN 1 ELSE 0 END) AS single_rbc_count,COUNT(CASE WHEN b.type = 'double rbc' THEN 1 ELSE 0 END) AS double_rbc_count,COUNT(CASE WHEN b.type = 'triple rbc' THEN 1 ELSE 0 END) AS triple_rbc_count FROM report r JOIN image_data i ON r.patient_id = i.patient_id LEFT JOIN  BLOB_TAB b ON i.image_id = b.img_id WHERE r.patient_id = ? GROUP BY r.patient_id;",
        [patient_id],
    )
    res = cur.fetchall()
    con.close()
    return res
