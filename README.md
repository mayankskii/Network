# Network
 Neural Network based product recommendation
 
 https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/8655105502604976/2769974586555098/7914230718189587/latest.html

 ```python
 from scipy.spatial.distance import cdist
 sub_arr = cdist(sira, sira, lambda u, v: u>=v)
 concern = (sub_arr*concern).sum(axis=0)
 fraud = (sub_arr*fraud).sum(axis=0)
 ```
%pip install optbinning pandas-profiling -U -q

import pandas as pd
import numpy as np

sdf = pd.concat([transformed_data, holding_matrix], axis=1)

from pandas_profiling import ProfileReport
profile = ProfileReport(sdf, title="Univariate Minimal Report", minimal=True)

profile.to_file("minimal_report.html")

import json
json_data = json.loads(profile.to_json())

keys = ['variable_name', 'n_distinct', 'p_distinct', 'is_unique', 'n_unique', 'p_unique', 'type', 'hashable', 'ordering', 'n_missing', 'n', 'p_missing', 'count', 
 'memory_size', 'n_negative', 'p_negative', 'n_infinite', 'n_zeros', 'mean', 'std', 'variance', 'min', 'max', 'kurtosis', 'skewness', 'sum', 
 'mad', 'range', '5%', '25%', '50%', '75%', '95%', 'iqr', 'cv', 'p_zeros', 'p_infinite', 'monotonic_increase', 'monotonic_decrease', 
 'monotonic_increase_strict', 'monotonic_decrease_strict', 'monotonic', 'top_ten_value_counts_without_nan', 'bottom_ten_value_counts_without_nan']

profile_dict = {k:[] for k in keys}
for i in json_data['variables'].keys():
    for j in keys:
        try:
            if j=='variable_name':
                profile_dict['variable_name'].append(i)
            elif j=='top_ten_value_counts_without_nan':
                top_ten_value_counts_without_nan = list(sdf[i].value_counts().iloc[:10].index.astype(str))
                profile_dict[j].append(top_ten_value_counts_without_nan)
            elif j=='bottom_ten_value_counts_without_nan':
                bottom_ten_value_counts_without_nan = list(sdf[i].value_counts().iloc[-10:].index.astype(str))
                profile_dict[j].append(bottom_ten_value_counts_without_nan)
            else:
                profile_dict[j].append(json_data['variables'][i][j])
        except:
            profile_dict[j].append(np.nan)

pd.DataFrame(profile_dict).to_excel('univariate summary.xlsx', index=False)

ph_profession
Managing Director
Art Director
Retired
Company Chairman
Company Director
Credit Broker
Property Manager
Insurance Broker
IT Manager
Housewife
Nurse
Chief Executive
Physiologist
Gallery Owner
Lawyer
Financial Consultant
Chartered Accountant
IT Consultant
Dental Surgeon
Judge
Consultant
Fund Manager
Commodity Broker
Bank Staff
Accountant
Finance Director
Investment Manager
Stockbroker
Chartered Surveyor
Film Producer
Auctioneer
Interior Designer
Commercial Manager
Caterer
Fashion Designer
Investment Banker
Operations Manager
Solicitor
Account Manager
Barrister
Sales Assistant
Creative Director
Actuary
Tax Consultant
Photographer
Marketing Director
Property Developer
Economist
Bank Manager
Equity Agent
Veterinary Surgeon
Marketing Consultant
Landowner
Independent Means
Research Scientist
Estate Manager
Farmer
Ship Broker
Haulage Contractor
Police Officer
Manager
Landlord
Hospital Doctor
General Practitioner
Artist
Teacher
Technical Director
Voluntary Worker
Management Consultant
Editor
Broadcaster
Unemployed
Publisher
Property Buyer
Writer
Recruitment Consultant
Queens Counsel
Doctor
Restaurateur
Underwriter
Financial Analyst
Proprietor
Actress
Architect
Product Manager
Business Consultant
Designer
Merchant Banker
Financier
Operations Director
Financial Advisor
Finance Manager
Hotelier
Surveyor
Assistant Teacher
Human Resources Manager
Finance Officer
Sawyer
Horse Trainer
Racehorse Groom
Journalist
Tennis Coach
Wine Merchant
Professional Footballer
Marketing Manager
Courier
Medical Consultant
Landscape Architect
TV Announcer
Teachers Assistant
Car Dealer
Advertising Staff
Administration Staff
Surgeon
Author
Entertainer
Auditor
Professional Sportsman
Project Manager
Professor
Gardener
Portfolio Manager
Investment Advisor
Researcher
Software Consultant
Agricultural Consultant
Sales Director
Human Resources Staff
Health Advisor
Medical Practitioner
Executive Officer
Business Analyst
Insurance Consultant
Insurance Underwriter
Account Director
Currency Trader
Engineer
Secretary
Civil Servant
Commodity Dealer
Jeweller
Account Executive
Farm Manager
Pilot
Motor Racing Driver
Public Relations Officer
House Parent
Company Secretary
Bursar
Historian
Contractor
Building Contractor
Motor Trader
Property Dealer
Furniture Dealer
Paediatrician
General Manager
Sportsman
Design Director
Legal Executive
Research Analyst
Design Manager
Warehouseman
Headteacher
Book-Keeper
Chemical Engineer
Plumber
Graphic Designer
Patent Attorney
Planning Officer
Photographer - Shop Owner
Metal Dealer
Civil Engineer
Film Director
Advertising Director
Fashion Photographer
Manager - Sports
Producer
Aeronautical Engineer
Therapist
Advertising Executive
Art Dealer
Dentist
Computer Programmer
Office Manager
Marketing Executive
Magistrate
Estate Agent
Payment Officer
Data Processing Manager
Casino Proprietor
Petroleum Engineer
Psychotherapist
Development Manager
Builders Merchant
Purchasing Manager
Acoustic Engineer
Landscape Gardener
Psychiatrist
Psychologist
Television Presenter
Fitness Instructor
Sales Manager
Accounts Assistant
Sports Commentator
Property Valuer
Audit Manager
Speech Therapist
Market Trader
Associate Director
Customs & Excise Officer
Yacht Master
Pensions Manager
Hairdresser
Chartered Engineer
Research Director
Marketing Agent
Driving Instructor
Medical Officer
Oil Broker
Law Clerk
Physiotherapist
Director/Company Director
Homecare Manager
Compliance Officer
Consultant Engineer
Pathologist
Lecturer
Footballer - Semi Professional
Town Clerk
Sales Executive
Opera Singer
Insurance Representative
Not In Employment
Landlady
Loss Adjustor
Mechanical Technician
Security Controller
Gynaecologist
Managing Clerk
Energy Analyst
Musician
Architects Technician
Neurologist
Buyer
Curtain Maker
Smallholder
Pharmacist
Radio Presenter
Publican
Television Producer
Anaesthetist
Conference Organiser
Member Of Parliament
Revenue Clerk
Importer
Software Engineer
Househusband
Veterinary Nurse
Chef
Sports Agent
Administrator
Building Engineer
Shipping Clerk
Coach Builder
Television Director
Charity Worker
Events Organiser
Builder
Scientist
Telecommunication Consultant
Bakery Assistant
Director of Planning
Quantity Surveyor
Bank Clerk
Optician
Sportswoman
Mortgage Consultant
Promoter
Personal Assistant
Mature Student - Living At Home
Houseman
Environmental Health Officer
Share Dealer
Tutor
Sculptor
Inspector - Insurance
Music Producer
Money Broker
Comedian
Electrical Contractor
Communications Officer
Rugby Player - Professional
Legal Advisor
Shop Manager
Environmental Consultant
Treasurer
Transport Manager
Education Advisor
Administration Assistant
Aircraft Cabin Crew
Electrical Engineer
Manufacturing Agent
Florist
Seamstress
Art Historian
Abstractor
Patent Agent
Letting Agent
Radio Producer
Fund Raiser
Biologist
Town Planner
Tree Surgeon
Golfer
Contract Manager
Cleaning Contractor
Gas Technician
Media Planner
Head of Trade
Building Surveyor
Furrier
Claims Manager
Safety Officer
Marketing Co-ordinator
Document Controller
Classical Musician
Construction Worker
Hotel Consultant
Research Consultant
Tax Advisor
Singer
Marine Broker
Pensions Consultant
Advertising Contractor
Local Government Officer
Nutritionist
Computer Manager
Masseur
Antique Dealer
Optometrist
Project Co-ordinator
Physician
Examiner
Art Critic
Song Writer
Orthopaedic Technician
Travel Consultant
Forester
Industrial Designer
Design Engineer
Agricultural Merchant
Technical Manager
Administration Officer
Security Consultant
Mechanic
Beauty Therapist
Translator
Administration Manager
Diplomat
Distribution Manager
Script Writer
Customer Advisor
Mining Engineer
Commissioned Officer
Tea Blender
Vicar
Trainer
Sheriff
Sales Representative
Loans Manager
Land Agent
Carpetfitter
Medical Secretary
Secretary And PA
Hair Stylist
Curator
Regulator
Training Consultant
Editorial Consultant
Training Advisor
Office Administrator
Chiropractor
Computer Analyst
Clerical Officer
Horticulturalist
Locum Pharmacist
Professional Boxer
Golf Club Professional
Book Seller
Developer
Shop Keeper
Surveyor - Chartered
Jeweller - refer
Project Engineer
Procurator Fiscal
Funeral Director
Notary Public
Astronomer
Furniture Remover
Assistant Caretaker
TV Editor
Shipping Officer
Jewellery Consultant
Barber
Welfare Officer
Gambler
Jockey
Library Manager
Sports Coach
Nursery Worker
Osteopath
Dock Pilot
Coin Dealer
Arbitrator
Textile Consultant
Technical Advisor
Garage Manager
Not Employed Due to Disability
Surgeon - N.H.S.
Production Planner
Salesman
Production Engineer
Hotel Worker
Shop Assistant
Meat Wholesaler
Guest House Proprietor
Actor
Landworker
Midwife
Investigator
Travel Agent
Roofer
Systems Manager
Music Teacher
Education Officer
Geologist
Employee
Agricultural Engineer
Applications Programmer
Advertising Manager
Pet Minder
Fairground Worker
Credit Controller
Homeopath
Dietician
Medical Technician
Computer Operator
Beautician
Council Worker
Fish Merchant
Tax Analyst
Carer - Professional
Upholsterer
Decorator
Stock Exchange Dealer
Student - Living at Home
Forensic Scientist
Mortgage Broker
Health Therapist
Professional Sports Coach
Sign Maker
Service Manager
Liaison Officer
Screen Writer
Bakery Manager
Aircraft Engineer
Fisherman
Occupational Therapist
Computer Consultant
Boat Builder
Postman
Quality Manager
Construction Engineer
Heating & Ventilation Engineer
Painter And Decorator
Printer
Product Designer
Pharmacy Manager
Flour Miller
Joiner
Medical Advisor
Insolvency Practitioner
Hospital Consultant
Quality Controller
Clothing Design Cutter
Records Supervisor
Geophysicist
Moneylender
Massage Therapist
Radiologist
Web Designer
Security Guard
Practice Manager
Medical Supplier
Cabinet Maker
Stationer
College Principal
Brewer
Tobacconist
Car Salesman
Building Manager
Mining Consultant
Valuer
Driving Instructor - Advanced
Loss Assessor
Heating/Ventilation Engineer
Scrap Dealer
Disc Jockey
Refrigeration Engineer
Diamond Dealer
Motor Mechanic
Furniture Restorer
Homeworker
Fuel Merchant
Model
Insurance Assessor
Probation Officer
Carer
Transport Consultant
Relocation Agent
Flying Instructor
Show Jumper
Advertising Agent
Project Leader
Composer
Publishing Manager
Yoga Teacher
Restorer
Golf Caddy
Watchmaker
Poultry Worker
Care Manager
Production Manager
Baker
Prison Officer
Youth Worker
Private Investigator
Counsellor
Revenue Officer
Archaeologist

brokerage
Roxburgh Group
G D Anderson & Co Ltd (Connect)
Marsh Private Clients (Letchworth)
Firth & Scott (Insurance Brokers) Ltd (Premier)
PIB Risk Services Ltd T/as PIB Insurance Brokers (Retford)
Irvine Commercial Insurance Brokers Ltd (Hedron Network)
Affinity Select Insurance Services Ltd (Advantage)
Hugh J. Boswell Limited trading as Saffron Insurance (Premier)
Lloyd Bolam Insurance Brokers Ltd (Brokerbility)
Smith England (Willis Network)
Aon (Cardiff) 606
CBC UK Ltd
A-Plan (RK Harrsion)
Aston Lark Limited
Lycett Browne-Swinburne & Douglass Ltd (Willis Nework)
Robins Row Ltd (Premier)
Robins Row Limited (Premier)
Gauntlet Insurance Services Ltd
Wesleyan Financial Services Limited
Arthur J. Gallagher (UK) Ltd (Fine Art, Museums & Exhibitions)
Anthony Wakefield & Company Limited
Arthur J Gallagher (UK) Ltd (London - Private Clients)
Arthur J Gallagher (UK) Ltd (Leicester)
Willis Limited (Ipswich)
T L Dallas & Co Ltd
Arthur J Gallagher (UK) Ltd (Leeds)
Towergate (Henley)
A-Plan (Howden UK, Birmingham)
Arthur J Gallagher (UK) Ltd (Belfast(O)
Sutton Winson Ltd
Lonmar Global Risks
Churchill Insurance Consultants
Verlingue (Redhill)
Lycett Browne-Swinburne & Douglass Ltd (Billingshurst) (Willis Nework)
Harold Wilson (Insurances) Limited
Verlingue
Todd and Cue Ltd (Brokerbility)
Marsh Private Clients (North)
A-Plan (Howden Insurance, London)
Carroll Holman Insurance Brokers
Alexander Miller
Lumley Insurance Limited
BHIB Limited t/as Brokerbility Insurance Brokers
Aon (HIBL Leeds)
Arthur J Gallagher (UK) Ltd (Belmont)
Oddie Dalton & Co Ltd
Berkeley Insurance Group UK Ltd (Cleo)
Clarke Dove (Insurance Brokers) Ltd (Cleo)
Arthur J Gallagher (UK) Ltd (Exeter - Stackhouse)
Mathews Comfort & Co Ltd (Cleo)
SPF Private Clients Limited
Arthur J Gallagher (UK) Ltd (Stackhouse Poole)
Web Shaw Ltd t/a Jacksons
David Oliver Associates
Towergate (Smith & Pinching GIS)
Arthur J Gallagher (UK) Ltd (Stackhouse Manchester)
Alan Boswell Insurance Brokers (Cambridge)
Alan Boswell Insurance Brokers (Norwich)
Towergate (Hull)
Bruce Stevenson Insurance Brokers Ltd
Kennett Insurance Brokers Ltd (Cleo)
ProAktive Ltd
Ault Insurance Brokers (Premier)
Towergate (Edinburgh & Borders)
R E Hutt & Company Ltd
Russell Scanlan Ltd
Arthur J Gallagher (UK) Ltd
James Hallam Limited
KGJ Commercial Insurance Services Limited
Marsh Commercial (Perth)
Ashley Page Insurance Brokers Limited (Hedron Network)
Ernest R Shaw Ltd
Tony McDonagh & Co Ltd (Cleo)
Abbott & Bramwell Ltd (Cleo)
Marsh Commercial (Aberdeen)
W B Baxter Ltd (Cleo)
Erimus Insurance Brokers
Arthur J Gallagher (UK) Ltd (Liverpool)
Arthur J Gallagher (UK) Ltd (Wakefield)
Arthur J Gallagher (UK) Ltd (Bollington Ins Brokers Ltd)
Towergate (Warwick)
Marsh Commercial Private Clients (PC Midlands)
Luker Rowe - Brokerbility
IFM Insurance Brokers Ltd
Arthur J Gallagher (UK) Ltd (Gloucester)
Backhouse Insurance Brokers Ltd
Caunce O’Hara Insurance Brokers Limited
Peter Hoare & Co Ltd (Brokerbility)
COBRA Uk & Ireland Ltd
Edinburgh Risk Management (General) Ltd
Marsh Commercial (Glasgow)
Arthur J Gallagher (UK) Ltd (Stackhouse Guildford)
Arthur J Gallagher (UK) Ltd (Manchester)
A-Plan (Alton)
Turner Rawlinson & Co Ltd (Cleo)
Guest Krieger Limited
Arthur J Gallagher (UK) Ltd (Guernsey)
Arthur J Gallagher (UK) Ltd (Birmingham)
David Roberts & Partners (Insurance Brokers) Ltd
John Bateman Insurance Consultants Limited
AIBL re Finch Commercial Insurance Brokers Limited (Southampton) CANX
Verlingue (Egham)
Marsh Commercial (Reading)
Arthur J Gallagher Services (UK) Ltd
Marsh Commercial (Bristol)
AIBL re Finch Commercial Insurance Brokers Limited (Reading)
M & N Insurance Services Ltd (Advantage)
J M Glendinning (Insurance Brokers) Ltd (Guiseley)
Towergate (London North)
Towergate (Stoke)
Hamilton Leigh Ltd
MFL Insurance Group Ltd  (Premier)
GS Group
Towergate (Taunton)
Higos Insurance Services Ltd
Thomas Carroll Private Clients Ltd CANX
MacDonald Group
Greenwood Moreland Insurance Brokers Ltd t.a Greenwood Moreland (Premier)
Abbey Bond Lovis Ltd
Alan R Mackay & Co Ltd ta Mackay Corporate Insurance Brokers (Brokerbility)
Lawrence Fraser Brokers
R A Cowen & Partners Ltd
Spence (Insurance Services) Ltd
J Hatty & Co
Marsh Commercial (Kirkwall)
Marsh Commercial (Oban)
Marsh Commercial (Dundee)
Towergate (Leicester)
Cox Mahon Limited
Ten Insurance Services Limited
Richmond House Insurance Brokers Limited (Hedron Network)
Arthur J Gallagher (UK) Ltd (IOM)
Lloyd & Whyte Ltd
Hettle Andrews & Associates Ltd
G A Hinks & Co Limited (Premier)
Alan Boswell Insurance Advisors (Peterborough)
McClarrons Limited (Premier)
Riskworks Business Services Ltd (Hedron Network)
OLD Insurefirst Limited T/A Bush And Associates
Lockton Companies LLP (Private Clients)
Griffiths & Armour (Liverpool)
Rollinson Smith & Company Limited (Premier)
Arthur J Gallagher (UK) Ltd (Bristol)
Stanhope Cooper
Rees Astley Insurance Brokers Limited (Cleo) Shrewsbury
Bennett Christmas Insurance Brokers Ltd (Premier)
Clegg Gifford & Co Ltd
Hiscox Private Client
One Broker (Cambridge) Ltd (Willis Network)
C Tarleton Hodgson & Son Ltd
James Hallam Ltd t.a Hallam Burgoyne
CIC Insurance Services Ltd
JPM Insurance Advisers Ltd (Hedron Network)
PIB Risk Services Ltd t.a PIB Insurance Brokers (Nottingham)
Marsh Private Clients (Edinburgh)
Arthur J Gallagher (UK) Ltd (Southampton)
Southall Harries Ltd (Hedron Network)
Walter A Wright Insurance Broker Ltd
Butterworth Spengler South West Ltd
UKGlobal Broking Group Limited
Gauntlet Risk Management Ltd (Willis Network)
County Insurance Consultants Ltd t/a Rawlins Insurance
Ellis David Ltd (Hedron Network)
Arthur J Gallagher (UK) Ltd (Belfast(G))
Willis & Company (Insurance Brokers) Ltd (Cleo)
Adler Fairways Insurance Brokers Limited t/as Adler Fairways
Towergate (Poole)
Corri Ltd
Allied Wessex Westinsure Ltd
PSP Insurance & Financial Solutions (Hedron Network) Chippenham
PIB Risk Services Ltd T/as PIB Insurance Brokers (Halifax)
Watkin Davies Insurance Consultants
Scrutton Bland Insurance Brokers (Willis Network)
Peter Hattersley & Partners Ltd (Advantage)
Christopher Trigg Ltd (Hedron Network)
Rowlands & Hames Insurance Brokers
H.J. Roelofs t a Bordengate Insurance (Willis Network)
M S Macbeth (Hedron Network)
GPS Insurance Brokers (Hedron Network)
A-Plan (High Wycombe)
W K Insurance Group (Cleo)
Henshalls Insurance Brokers
Square Mile Insurance Services Ltd
Marsh Commercial (Elgin)
Arthur J Gallagher (UK) Ltd (Stackhouse High Wycombe)
OLD DCJ Group Insurance & Risk Management Limited
Saffron Insurance Services Limited (Premier)
MCM Insurance Services (Premier)
Aston Lark Limited t/as Absolute Products Ltd
A.I.P.S (Advantage)
Towergate (Leeds)
Thompson and Richardson Ltd (Sleaford) (Premier)
PIB Risk Services Ltd T/as PIB Insurance Brokers (Brighouse)
PIB Risk Services Ltd T/as PIB Insurance Brokers (Gloucester)
*DO NOT USE* Green Insurance Brokers Limited t/a Reid Briggs OLD
Grove & Dean Ltd tas Performance Direct (Advantage)
Kelvin Smith (Insurance Brokers) Ltd
County Insurance Consultants Ltd t/as Inspire Risk Management CANX
PSP Insurance & Financial Solutions (Hedron Network) Saltash
Marsh Private Clients- Collectors scheme
Partners& (Oxford)
Champion Insurance Brokers
James & Lindsay Ltd (Premier)
Reich Insurance Brokers Limited
Heath Crawford & Foster Ltd (Advantage)
SEIB Insurance Brokers Limited t/a Lansdown Insurance Brokers
Signature Insurance Services Limited (Hedron Network)
Crompton Bailey Ltd (Cleo)
Professional Mortgage Services UK LLP
Oakwood (Insurances) Limited
Affinity Brokers Ltd (BNL)
Stonebridge Corporate Ltd (Hedron Network)
Profile Insurance Services Ltd (Advantage)
Borland Insurance Ltd (Hedron Network)
Romero Insurance Brokers Ltd
Heath Crawford Financial Services LLP (Hedron Network)
Robins Row Limited (Bognor Regis) (Premier)
Magnet Insurance Services Ltd (Hedron Network)
Kerr Henderson (General Insurance Services) Ltd
Allbright Bishop Rowley Limited (Hedron Network)
Andrew Thompson & Associates Ltd
Coversure Insurance Services Ltd
Calcluth & Sangster (Insurance Brokers) Ltd
Ainleys Insurance Consultants (CANX)
Merritt Insurance Services Ltd
A Manning UK Limited (Hedron Network)
Aston Lark Limited (Caterham)
Arthur J Gallagher (UK) Ltd (Llantrisant)
Ravenhall Risk Solutions Ltd
BPS Broking Desk (Hedron Network)
Aston Lark Limited (Padstow)
Bruce Stevenson Insurance Brokers (Glasgow)
Hazelton Mountford Ltd
Lockton Companies LLP
James Brown & Sons (Somerset) Ltd
One Broker (GDIS) (Willis Network)
Yutree Insurance Ltd t.as Yutree Underwriting
*DO NOT USE* GRP Retail Ltd t/a Marshall Wooldridge Ltd OLD
Lift Insurance (Hedron Network)
Gomm (2000) Ltd (Premier)
Sutcliffe & Co (Hedron Network)
Leagold Ltd
Abbeyfields Insurance (Wedding Insurance Group)
Simmons Gainsford Insurance Solutions Ltd
Marsh Commercial (London)
Headley Group Ltd TA Nugent Debenham (Premier) CANX
Pyke Smith & Cutler Ltd (Advantage)
Guy Penn & Company Ltd (Premier)
OLD Guardian IB
SG Busby Insurance Brokers (Advantage)
Illingworth Insurance & Financial Services ((Advantage)
A-One Insurance Services (BMTH) Ltd
Aston Lark Limited (Farnborough)
Dennis Watkins & Co Ltd (Premier)
Affinitive Insurance Brokers Ltd
PSC UK Insurances Ltd t/a Turner Insurance Group
Weatherbys Hamilton LLP
WTJ Insurance Brokers Ltd
Robert Gerrard & Co Ltd (Premier)
HFIS Ltd t/a Hamilton Fraser Insurance Solutions
Howe Maxted Group (Hedron Network)
A-Plan Insurance (Wilmslow)
Williamson Carson Insurance Brokers (Photographers scheme)
PSC UK Insurances Ltd t/a Absolute Insurance Brokers
Castleacre Insurance Services Ltd
C.C Flint and Company Limited
Arthur J Gallagher (UK) Ltd (Edinburgh OPM)
A-Plan (RK Harrison, Oak transfer)
Blue Rock Insurance Brokers Ltd
Home Counties Insurance Services Limited
PSC UK Insurances Ltd t/a Abaco Insurance Brokers (Cleo)
CoverMy Ltd (Hedron Network)
505 Referrals (HPC)
Movo Partnerships Ltd
Nowell and Richards Insurance Services Limited
SJL (Worcester) Ltd t/as SJL Insurance Services
ICAEW (HPC)
First Insurance Solutions Ltd
Hugh J Boswell (Premier)
Marsh Commercial (Carlisle)
Marsh Commercial (Witham)
Marsh Commercial (Kendal)
Marsh Commercial (Penrith)
Marsh Private Clients (PC South)
Grosvenor Chester Ltd (Hedron Network)
County Insurance Consultants Ltd t/a Sagar Insurances
Hallett Independent Limited (Commercial)
Quote Four Ltd (Premier)
C & C Insurance Brokers Ltd
Hallett Independent Limited (Binder)
*DO NOT USE* Green Insurance Brokers Ltd OLD
Kingsbridge Risk Solutions Ltd
Airsports Insurance Bureau Limited
Hudson Foster LLP
Ashbourne Insurance Services (Hoddesdon) Ltd (Hedron Network)
OLD Allcover T/a LDS Associates & Re
Chambers and Newman Ltd
Arthur J Gallagher (UK) Ltd (HR Owen)
Sovereign Insurance Services Group (HPC)
Stanmore Insurance Brokers Limited
BJP Insurance Brokers Ltd
Jensten Insurance brokers Ltd (Nottingnam)
T L Dallas & Co (NI) Ltd
Redwood Business Insurance Services Ltd (Premier)
Pace Ward Ltd (Premier)
Dickson Financial Services Ltd t/as Innovation Broking
Cheviot Insurance Services
John Morgan Partnership Ltd
Aston Lark Limited (Colchester)
PIB Risk Services Ltd T/as PIB Insurance Brokers (Bristol)
PIB Risk Services Ltd T/As PIB Insurance Brokers (Leicester)
PIB Risk Services Ltd T/As PIB Insurance Brokers (Leicester) CANX
Momentum Broker Solutions Ltd
Arthur J Gallagher (UK) Ltd (Portmore IB Ltd)
JPM Insurance Management Ltd trading as BEAM
LRG Financial Solutions Ltd T/a LRG Insurance
Blythin & Brown Insurance Brokers Ltd
Berns Brett Ltd
PIB Risk Services Ltd T/as PIB Insurance Brokers (London)
Vizion Insurance Brokers Limited (Hedron Network)
Arthur J Gallagher (UK) Ltd (Hughes Plane Ltd)
Hallett Independent Limited (Private Client)
Tysers (Hitchin – Private Client)
Tysers (London – Private Client)
Everett Mead Limited
GRP Retail Limited
Clarke Williams Ltd
Giles Gowers Insurance Associates
Porterhouse Brokers LLP
Towergate (Cardiff)
Higos Thatch Scheme
Hartinsure Ltd t/a Hart Insurance Brokers
Marlow Gardner & Cooke Ltd (Brokerbility)
Arthur J Gallagher (UK) Ltd 7000064 UK (Hiscox Underwriting Limited)
Nsure Limited (Hedron Network)
One Broker (Norwich) Ltd (Willis Network)
ACM Broking Ltd
Berkeley Alexander (Specialist)
Jaggi & Company Ltd
Weald Insurance Brokers Ltd (Premier)
OLD Alan & Thomas Insurance Brokers Limited (Open Market)
Towergate (Hemel)
Towergate (Private Clients - Hiscox Colchester)
Hayes Parsons Ltd
Robison & Co (Hedron Network)
CCVRS t/a Towergate (Jersey)
Centor Insurance & Risk Management Limited
Marsh Brokers Ltd Ta Private Clients
MPW Insurance Broker Ltd (Brokerbility)
Willis Towers Watson
R A Rossborough (Insurance Brokers) Ltd
Towergate (Sevenoaks)
Berns Brett (Maidenhead Region)
OLD BHK Insurance Services Ltd
J Bennett & Son (Insurance Brokers) Ltd
Knighthood Corporate Assurances Services plc (Premier)
Brownhill Insurance Group Limited (Willis Network)
Bryan Baines t/as Coversure.co.uk
Butterworth Spengler Insurance Group
Tysers (Manchester – Private Client)
Barts Insurance Brokers Limited (Hedron Network)
DPi Bros Ltd t/a DPI Insurance (Hedron Network)
Towergate (Bury St Edmunds)
County Insurance Consultants Ltd t/as IDouglas Insurance Brokers Ltd
Ryan Insurance Group Limited
Prospero Insurance Brokers Ltd (Premier)
County Insurance Consultants Ltd trading as CJN Insurance Services
Provenance Insurance Brokers
OLD Alan & Thomas Insurance Brokers Basingstoke Ltd
Bernard Saxon Insurance Services (Premier) t/a BSIS
Lycett Browne-Swinburne (Berwick St Leonard)(Willis Nework)
J & V Risk Solutions Ltd (Hedron Network)
Jensten Insurance Brokers (Ripley)
Ascend Broking Group Limited (Cleo)
G & C Insurance Services Limited t/a We'reSure
Packetts
Insurance Services (Surrey) Ltd (Premier)
Eggar Forrester Insurance Limited
Bullerwell & Co Ltd
PIB Risk Services Ltd T/as PIB Insurance Brokers (York)
Clear Insurance Management Limited
Citygate Insurance Services Ltd (Advantage)
Protect Commercial Insurance Solutions Ltd
D B Wood Ltd
Arthur J Gallagher (High Wycombe)
Town and Country Financial Services (UK) Ltd (Advantage)
Tyser & Co Ltd
Callaway and Sons Insurance Consultants Ltd (Hedron Network)
Arthur J Gallagher (UK) Ltd (Stackhouse London)
Finch Commercial Insurance Brokers t/a Miller & Co (Insurance Brokers) Ltd CANX
Specialist Individual Insurance Broking Ltd
Towergate (Glasgow)
Miles Smith Limited
Broadway Broking Group Limited
Tysers (Colchester - Private Client)
County Insurance Services Limited
Prescott Jones Ltd
Ashbourne Insurance Services Ltd (Hedron Network)
Arthur J Gallagher (UK) Ltd (Stackhouse South Woodham Ferrers)
David Roberts & Partners (York) Ltd
Horner Blakey Ltd (Hedron Network)
Schofield Insurance Brokers Ltd (Premier)
Consort Insurance Limited
Blackford Group Ltd
Daines Kapp Insurance Brokers Ltd (Cleo)
Towergate (Manchester)
Cover-Rule Limited t/a Ascott Insurance (Advantage)
Marsh Commercial (Southampton)
Boyd & Company Limited (Premier)
Forum Finance Ltd t/as Forum Insurance (Connect)
Richard Thompson (Insurance Brokers) Ltd
Hughes Insurance Services Limited
Cairn Corporate Ltd
W Denis Insurance Brokers Plc
Brunel Insurance Brokers Ltd (Hedron Network)
Hencilla Canworth Ltd
Arthur J Gallagher (UK) Ltd - Allison & Partners Household
SPF Oak
BLW Insurance Brokers Ltd
Aston Lark Limited (Derby)
Towergate (Wales and Bristol)
Miller Insurance Services Ltd
Sense Risk Solutions Limited (Willis Network)
John Henshall ltd t/as Bayliss & Cooke (Willis Network)
Thompson & Co (Risk Solutions) Ltd (Hedron Network)
Cobra Network Ltd
Attis Insurance Brokers Ltd (Premier)
Finch Commercial Insurance Brokers Limited t/as Finch Hughes & King (Premier) CANX
Lansdowne Woodward Ltd
Surrey Independent Advisers Ltd (Connect)
D.M. Cager (Insurance Brokers) Ltd
Movo Eastbounre Ltd CANX
Aston Lark Limited t/as Plester Group
Towergate (West Midlands)
Mint Insurance Brokers Ltd
Partners&. (South East)
Aston Lark Limited t/as Venture Insurance Brokers Ltd
A-Plan (Bournemouth)
Arthur J Gallagher (UK) Ltd (Chelmsford)
Hallsdale Insurance Brokers Ltd (Willis Network)
KPTI Ltd T/a IS Insurance Solutions (Hedron Network)
Marsh (Central Insurance Services)
Towergate (Kings Lynn)
A Plan (Bedford)
Mantra Consultancy and Capital Limited t/a Mantra Insurance (Willis Network)
Tysers (E&S)
Marsh Commercial (Inverness)
Towergate (Darlington)
Channel Insurance Brokers Limited
Onyx Brokers Ltd t/as Churchside Insurance (Hedron Network)
Blackmore Borley (Premier)
Inval Holdings Ltd t as Routen Chaplin
Robert Arthur Davies Limited (Advantage)
Anami Agencies Ltd (Premier)
Coversure Insurance Services Ltd (London East)
Arthur J Gallagher (UK) Ltd (Broker One Ltd)
K D Insurance Brokers (Hedron Network)
Drayton Ins. Limited (Premier)
T H March & Co Ltd
Hedron Connect Placement Desk
Channel Insurance Brokers (Jersey) Limited
Lycetts (HPC)
RS Risk Solutions Limited (Hedron Network)
Edison Ives Ltd (Hedron Network)
Towergate (Aberdeen)
Denis O Brown & Associates Limited
Wrightsure Services Ltd
Rockland Risk Services Limited t/a IGG Insurance Brokers
Vista Insurance Brokers Limited
Arthur J Gallagher (UK) Ltd (Bollington Underwriting Ltd)
Marsh FINPRO Scotland
Stewart Miller McCulloch & Co (Insurance Brokers) Limited t/as Classic Insurance Services
Johnstone Insurance Brokers Ltd (Premier)
T Oscar &Co Ltd  t/a Rollins Insurance Brokers (Hedron Connect)
MGN (Rugby) Ltd t/as Coversure Insurance Services (Rugby)
Headley Group Ltd CANX
Aston Lark (Bingley)
S & J Palmer t/as Coversure Insurance Services (Maidstone)
Park Insurance Services
Besso Limited
Specialist Risk Insurance Solutions Limited
Jensten Insurance brokers Ltd (Midlands)
*Do not use* Towergate (Personal Lines)
Arthur J Gallagher Ltd (Chester)
Radius (I.B.) Limited (Premier)
Bartlett & Co Ltd
Konsileo (Trading) Ltd
EIC Insurance Services Ltd (Premier)
Marsh Commercial (Edinburgh)
NW Risk Solutions Limited
Greenfield Insurance Services Limited (Premier)
Five Insurance Brokers Limited
OLD GRP Retail t/a R F Broadley
Choice Insurance Agency Ltd
W H & R McCartney Ltd T/A W H & R McCartney Insurance Brokers
Arthur J Gallagher (UK) Ltd (Isle of Wight)
James Hallam Limited t/as Pinner Risk Solutions
Kerry London Ltd (Isleworth)
EIG Limited
A.M.P. Insurance Brokers Limited
Willis (London Market)
LMR Insurance Services Ltd
PIB Risk Services Ltd T/as PIB Insurance Brokers (Glasgow)
Marsh Commercial (Swindon)
Sydney Charles Financial Services Limited
McCarron Coates Ltd (Premier)
JJ Yates & Co Ltd
AIBL re Finch Commercial Insurance Brokers Ltd (The Broker Network - Premier) CANX
Grayside Insurance Brokers (Premier) CANX
RIB Group Ltd t/as Rotherham Insurance Brokers
UBT (EU) Ltd t/a UBT Protect (Willis Network)
Perry Appleton Risk Services Ltd (Premier)
Hamilton Robertson Insurance Brokers Limited
CCRS Brokers Limited
Wilson Insurance Services Ltd t/as Coversure Insurance Services (Colne)
Oyster Risk Solutions
County Insurance Agencies Ltd (Hedron Network)
Towergate (Ipswich)
Marsh Commercial (Brighton)
Base Ins Brokers Ltd (Premier)
Howden Insurance Brokers Ltd (OM)
Coversure Insurance Services LTD (Hessle)
PIB Risk Services Ltd T/as PIB Insurance Brokers (Birmingham LRC)
J Safra Sarasin Brokerage Limited
County Insurance Consultants Ltd t/as JSW Insurance Services
Weir Insurance Brokers Limited
PIB Risk Services Ltd T/As Element Hinton (Insurance Brokers)
T&R Direct Ltd
AIBL re Finch Commercial Insurance Brokers Limited (Basingstoke) CANX
Saxon Insurance Brokers Limited
Jensten Insurance brokers Ltd (Kent)
Elevate Insurance Brokers Ltd
Clear Insurance Management Ltd (Leamington)
J M Glendinning (Insurance Brokers) Ltd (Newcastle)
Gen2 Broking Ltd (Premier)
WM Brokers Limited
Warwick Davis (Insurance Consultants) Ltd
Mercari Risk Management Ltd (Premier)
GRP Retail Ltd t/a Marshall Wooldridge Ltd
Aston Lark (Birmingham)
HISL Brokers Limited
Towergate (Private Clients - Hiscox Birmingham)
Coversure Insurance Services (Falkirk)
Paterson Risk Management Limited  (Premier)
Joseph W. Burley and Partners (UK) Ltd t/as Radius Insurance Solutions (Connect)
GRP Retail Limited t/a  Green Insurance Brokers Hub (Bexhill, Reid Briggs, Brighton)
PIB Risk Services Ltd T/As Arlington Insurance Services
FAM UK Consultancy Ltd /as Coversure Insurance Services Guildford
GRP Retail Ltd t/a Allcover (t/as LDS Associates) CANX
Towergate (Private Clients - Hiscox Maidenhead)
County Insurance Consultants Ltd
Towergate (Lewes)
Thomas Carroll (Brokers) Ltd
NBJ London Markets Limited
Chorley Broking Limited t/as Coversure Insurance Services (Chorley)
PIB Risk Services Ltd T/as PIB Insurance Brokers (Houghton)
PROSURA LTD (Premier)
Smith Robinson Limited
Coversure Insurance Services (Ilfracombe) & Coversure Insurance Services (Taunton)
Alastair James Insurance Brokers Ltd
The Bletchley Group Ltd
County Insurance Consultants Ltd t/a George Williams Insurance
Hampden & Co (HPC)
Coversure Bristol Limited (Tewkesbury)
Monaco Insurance Services Limited
Jensten Insurance brokers Ltd (Manchester)
Lloyd & Whyte Community Broking
Aston Lark Limited (Exeter)
Alan R Mackay & Co Ltd ta Mackay Corporate Insurance Brokers (Brokerbility) (Aberdeen)
Barkdene Ltd T/A Henry Seymour & Co
Keighley Broking Services Limited t/as Coversure Insurance Services (Keighley)
Edwards Insurance Brokers (Premier)
Independent Broking Solutions
R A Gary Consultancy Ltd t/as Coversure Insurance Services (Reading)
A M B Insurance Services Ltd (Premier)
Magus GI Limited t/as Coversure Insurance Services (Westminster)
GRP Retail Ltd t/a Bush & Associates
Clegg Gifford & Co Limited
GRP Retail Ltd t/a Britannia Consultants
Delta Corporate Risk Limited (Premier)
Bickley Insurance Services Ltd (Premier)
MAC Commercial and Professional Risks Limited
Marsh Commercial (Worcester)
Colmore Insurance Brokers Ltd (Premier)
Advanta Risk Ltd CANX
Partners&. (Midlands)
Endsleigh Insurance Services Limited
Ten Insurance Services Ltd (Leeds)
Insurewise Limited (Premier) CANX
Nationwide Broker Services Ltd
Thompson & Richardson (Lincoln) Ltd (Premier)
Towergate (Newcastle)
La Playa Limited CANX
One Broker Ltd (Uttings) (Willis Network)
JRT Insurance Broker Ltd CANX
Glentworth Portishead Limited (Marsh Pro) CANX
Marsh Commercial (Manchester)
Towergate Risk Solutions (Manchester) CANX
Jelf Insurance Brokers Ltd Harrogate APC CANX
Marsh Commercial (Hull)
Towergate (Galashiels) CANX
BCS Hendricks Ltd CANX
Towergate Dawson Whyte (Belfast)
Eastwood & Partners Limited
Aon (HIBL Kirmington)
Portcullis Insurance Brokers Limited (Cleo) CANX
Oakland Insurance Services Ltd CANX
Towergate (Norwich)
Towergate (Redruth)
Bridges Insurance Brokers Ltd (Willis Network) CANX
Munro & Sons Ltd t.as Munro Insurance Consultants CANX
Towergate (Contact Centre) CANX
Trust Insurance Group Services Ltd (Hedron Network) CANX
PSP Insurance & Financial Solutions (Hedron Network) Torquay CANX
J.N. Dobbin Limited t/as MRIB Group CANX
Alan Blunden & Co Ltd (Advantage) CANX
Jim Kelly & Co Limited (Hedron Network) CANX
Sole Bay Insurance Brokers
COBRA Insurance Brokers Ltd t/as COBRA GDK
Lycett, Browne-Swinburne & Douglass Limited t/as Robertson-McIsaac Insurance Brokers (Willis Nework)
Neil Willies Insurance Brokers Ltd (Hedron Network)
Gott and Wynne Limited (Premier) CANX
CCH Insurance Brokers Ltd (Premier) CANX
Insure Smart Limited (Advantage) (CANX)
Chubb Insurance Brokers Limited CANX
Towergate (Northampton)
ADVANCE INSURANCE SERVICES (MIDLANDS) LTD (Hedron Network) CANX
Beaumont Lawrence & Co Ltd (Hedron Network)
Russell Meers and Gill (Worcester) Limited (Hedron Network) CANX
HMS Insurance (Advantage) CANX
Castle Sundborn Ltd (Premier) CANX
Dixons Commercial Insurance Brokers (Hedron Network) CANX
Policywise Ltd
Arthur J Gallagher (UK) Ltd (Symmetry)
Riverdale Business Solutions Ltd t/as Riverdale Insurance (Premier) CANX
Tower Insurance Brokers Ltd (Willis Network)
Insure Risk Limited CANX
Gothic Insurance Brokers Ltd ta Certis Insurance Brokers (Premier) CANX
Burrow Humphreys Ltd (Advantage) CANX
Business Insurance Broking Services (Hedron Network) CANX
Reich Magen HNW Scheme
PH7 Insurance Brokers (Advantage) CANX
Jane Chewins Limited (Premier)
Woodward Markwell Ins Brks Ltd (Cleo)
OLD Real Insurance Brokers
ZZZZZZ- Gomez
Anthony James Insurance Brokers Ltd (Premier)
Finch Commercial Insurance Brokers Ltd t/a Citymain Insurance (The Broker Network - Premier) CANX
Rickard Lazenby International Limited (Hedron Network) CANX
Jelf Insurance Brokers Ltd Edinburgh
Towergate Dawson Whyte (Larne)
Senior Wright Limited
Abbey Bond Lovis Ltd (CANX)
D2 Corporate Solutions (BI)
Abbey Bond Lovis Ltd CANX
The Alan Stevenson Partnership Limited
Attis Insurance Brokers Ltd (Brigg) (Premier)
Lockton Companies International Ltd (Birmingham)
Brady Insurance Services Ltd
GS Group (Glasgow)
A Plan (Birmingham)
North East Insurance Services Ltd t as Coversure Newcastle
AIBL t/as Lockyers
A-Plan (RK Harrison, Dual transfer)
OLD County Insurance Consultants Ltd Trading as Britannia Consultants
Exchequer Risk Management Limited
County Insurance Consultants Limited trading as Anderson Ashcroft
J M Glendinning (Insurance Brokers) Ltd (North Yorkshire)
J.E. Sills & Sons Ltd
James Hallam Ltd
Gravity Risk Services Ltd
Towergate (London City)
AIBL re Finch Commercial Insurance Brokers Ltd (Premier) CANX
Prima Financial Services Ltd (Hedron Network)
SEIB Insurance Brokers Limited (Premier)
Gauntlet Insurance Services
Peter Lole & Co Ltd (Advantage) CANX
H W Wood Ltd
A-Plan (RK Harrison, Knight Frank)
Towergate Riskline (IBA 5) (Maidenhead)
Partners& (South Molton)
Coversure Insurance Services Middlesbrough T/A Coversure Insurance Services
County Insurance Consultants Limited t/as Wrexham Insurance Services
Astute Insurance Solutions Ltd
Brents of Brentwood Ltd (Advantage)
Eastlake & Beachell Limited (Advantage) Open market
Morgans Insurance Limited (Hedron Network)
Aon Limited (Isle of Man) (Glasgow APC) CANX
Quartz Insurance Brokers Limited
Risk Management and Insurance Services
![image](https://github.com/mayankskii/Network/assets/43214641/e1e37e51-b876-4137-b675-d3a8f32c7bb1)

